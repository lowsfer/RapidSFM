/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include "bruteForceMatcher.h"
#include <Runtime.hpp>
#include "Builder.h"
#include "Config.h"
#include <FiberUtils.h>
#include "casHash/cascadeHash.h"

#define USE_DELEGATED_STREAM_SYNC 1

namespace rsfm
{
namespace
{
inline bool canUseSqrNormPreComp()
{
    static const bool flag = [](){
        cudaDeviceProp prop;
        cudaCheck(cudaGetDeviceProperties(&prop, getCudaDevice()));
        return prop.major >= 8 || (prop.major == 7 && prop.minor >= 5);
    }();
    return flag;
}
inline size_t alignUp(size_t size) { return roundUp(size, size_t{128UL}); }
size_t getPreCompDeviceScratchNbBytes(uint32_t nbDesc, bool useCasHash)
{
	if (!useCasHash) {
		return 0UL;
	}
	const size_t biasNbBytes = alignUp(sizeof(cudapp::KArray<int32_t, 64>));
	const size_t bucketIndicesNbBytes = alignUp(sizeof(cudapp::KArray<uint8_t, cashash::nbScheme>) * nbDesc);
	const size_t histAndBuckerBegin = alignUp(sizeof(cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>) * 2);
	return biasNbBytes + bucketIndicesNbBytes + histAndBuckerBegin;
}
size_t getPreCompPinnedScratchNbBytes(bool useCasHash)
{
	return useCasHash ? 4UL : 0UL;
}
size_t getPreCompNbBytes(uint32_t nbDesc, bool useSqrNorm, bool useFourBit, bool useCasHash)
{
    const size_t transDesc4bSize = alignUp(useFourBit ? sizeof(typename Sift4bDescTraits::Descriptor) * nbDesc : 0UL);
    const size_t sqrNormSize = alignUp(useSqrNorm ? sizeof(uint32_t) * nbDesc : 0UL);
	const size_t casHashBucketEnds = alignUp(useCasHash ? sizeof(cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>) : 0UL);
	const size_t casHashBuckets = alignUp(useCasHash ? sizeof(uint32_t) * nbDesc * cashash::nbScheme : 0UL);
    return transDesc4bSize + sqrNormSize + casHashBucketEnds + casHashBuckets;
}
void preCompForMatch(std_byte* dst, const typename SiftDescTraits::Descriptor* src, uint32_t nbDesc,
    bool useSqrNorm, bool useFourBit, bool useCasHash,
	uint64_t& accNbDesc, cudapp::KArray<uint64_t, 128>& acc,
	const cudapp::KArray<int32_t, 64, 32>& casHashWeights, const cudapp::KArray<int32_t, 64>& casHashBiasBase, // must be on device memory
	std_byte* pinnedScratch, std_byte* deviceScratch, std::function<void(uint32_t)> maxBucketSizeHandler,
	cudaStream_t stream)
{
    ASSERT(isPtrAligned<16>(dst));
    if (useFourBit) {
        using Descriptor = typename Sift4bDescTraits::Descriptor;
        launchSiftDesc8bTo4b(reinterpret_cast<Descriptor*>(dst), src, nbDesc, stream);
        dst += alignUp(sizeof(Descriptor) * nbDesc);
    }
    if (useSqrNorm) {
        launchPreCompSiftSqrNorm(reinterpret_cast<uint32_t*>(dst), src, nbDesc, stream);
		dst += alignUp(sizeof(uint32_t) * nbDesc);
    }
	if (useCasHash) {
		auto* const casHashBucketEnds = reinterpret_cast<cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>*>(dst);
		dst += alignUp(sizeof(cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>));
		auto* const casHashBuckets = reinterpret_cast<uint32_t*>(dst);
		dst += alignUp(sizeof(uint32_t) * nbDesc * cashash::nbScheme);

		uint32_t* const pinnedMaxBucketSize = reinterpret_cast<uint32_t*>(pinnedScratch);

		auto* const bias = reinterpret_cast<cudapp::KArray<int32_t, 64>*>(deviceScratch);
		deviceScratch += alignUp(sizeof(cudapp::KArray<int32_t, 64>));
		auto* const bucketIndices = reinterpret_cast<cudapp::KArray<uint8_t, cashash::nbScheme>*>(deviceScratch);
		deviceScratch += alignUp(sizeof(cudapp::KArray<uint8_t, cashash::nbScheme>) * nbDesc);
		auto* const histAndBounds = reinterpret_cast<cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>*>(deviceScratch);
		deviceScratch += alignUp(sizeof(cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>) * 2);
		auto* const histogram = histAndBounds;
		auto* const idxAllocator = histAndBounds + 1;

		const bool isFirst = accNbDesc == 0;
		cashash::accumuateDesc(acc, nullptr, isFirst, src, nbDesc, stream);
		accNbDesc += nbDesc;
		cashash::computeBias(bias[0], casHashWeights, casHashBiasBase, acc, accNbDesc, stream);
		cashash::computeHashForBucketOnly(casHashWeights.data, bias->data, src, nbDesc, bucketIndices, nullptr, stream);
		cashash::buildHistogram(histogram, bucketIndices, nbDesc, stream);
		cashash::computeBucketBound(*idxAllocator, *casHashBucketEnds, pinnedMaxBucketSize, *histogram, stream);
		cashash::buildBuckets(bucketIndices, nbDesc, *idxAllocator, casHashBuckets, stream);
		launchCudaHostFunc(stream, [handler{std::move(maxBucketSizeHandler)}, pinnedMaxBucketSize](){handler(*pinnedMaxBucketSize);});
	}
}
} // unnamed namespace

BruteForceMatcher::BruteForceMatcher(Builder& builder)
    : mBuilder{builder}
    , mUseSqrNormPreComp{canUseSqrNormPreComp()}
    , mUseSift4b{builder.config().matcher.useSift4b}
	, mUseCascadeHash{builder.config().matcher.method == Config::Matcher::kCascadeHash}
{
    if (mUseSift4b) {
        // only sqrNorm version is implemented for Sift4b.
        ASSERT(mUseSqrNormPreComp);
    }
	if (mUseCascadeHash) {
		ASSERT(mUseSqrNormPreComp && !mUseSift4b); // 4b is not implemented for cascade hash
		mDescAcc = allocCudaMem<cudapp::KArray<uint64_t, 128>, CudaMemType::kDevice>(1);
		mHostScratch = allocCudaMem<std_byte, CudaMemType::kPinned>(getPreCompPinnedScratchNbBytes(mUseCascadeHash));
		mCasHashWeights = allocCudaMem<cudapp::KArray<int32_t, 64, 32>, CudaMemType::kDevice>(1);
		mCasHashBiasBase = allocCudaMem<cudapp::KArray<int32_t, 64>, CudaMemType::kDevice>(1);

		std::vector<int8_t> weightData(64*128);
		std::normal_distribution<float> weightDist{0.f, 32.f};
		std::default_random_engine rng{11u};
		std::generate(weightData.begin(), weightData.end(), [&](){
			return int8_t(round(clamp(weightDist(rng), -128.f, 127.01f)));
		});
		std::vector<int32_t> biasData(64);
		std::normal_distribution<float> biasDist{0.f, 16.f};
		for (int i = 0; i < 64; i++) {
			float v = 0;
			for (int j = 0; j < 128; j++) {
				v += weightData[i * 128 + j] * clamp(biasDist(rng), -64.f, 64.f);
			}
			biasData[i] = int32_t(round(v));
		}
		cudaCheck(cudaMemcpyAsync(mCasHashWeights.get(), weightData.data(), sizeof(mCasHashWeights[0]), cudaMemcpyHostToDevice, mStream.get()));
		cudaCheck(cudaMemcpyAsync(mCasHashBiasBase.get(), biasData.data(), sizeof(mCasHashBiasBase[0]), cudaMemcpyHostToDevice, mStream.get()));
		launchCudaHostFunc(mStream.get(), [w{std::move(weightData)}, b{std::move(biasData)}](){});
	}
}

BruteForceMatcher::~BruteForceMatcher()
{
    std::lock_guard lk{mLock};
    for (auto& item : mPreCompDataKeys) {
        if (item.second != cudapp::storage::kInvalidKey) {
            mBuilder.storageManager().removeItem(item.second);
            item.second = cudapp::storage::kInvalidKey;
        }
    }
}

bool BruteForceMatcher::needPreCompStorage() const {
    return mUseSqrNormPreComp || mUseSift4b || mUseCascadeHash;
}

void BruteForceMatcher::registerImage(ImageHandle hImage, const typename BruteForceMatcher::Descriptor* desc, uint32_t nbDesc, cudaStream_t userStream)
{
    std::lock_guard<fb::mutex> lk{mLock};
    if (userStream != mStream.get()) {
        connectStreams(userStream, mStream.get(), mEvent.get(), &mEventLock);
    }

	const cudaStream_t stream = mStream.get();

	const size_t requiredDeviceScratchSize = getPreCompDeviceScratchNbBytes(nbDesc, mUseCascadeHash);
	if (requiredDeviceScratchSize > mDeviceScratchSize) {
		cudaCheck(cudaStreamSynchronize(mStream.get()));
		mDeviceScratch = allocCudaMem<std_byte, CudaMemType::kDevice>(requiredDeviceScratchSize);
		mDeviceScratchSize = requiredDeviceScratchSize;
	}

    const size_t preCompSize = getPreCompNbBytes(nbDesc, mUseSqrNormPreComp, mUseSift4b, mUseCascadeHash);
    if (!needPreCompStorage()) {
        ASSERT(preCompSize == 0);
        return;
    }
    auto preComp = mBuilder.cudaMemPool<CudaMemType::kDevice>().alloc<std_byte>(preCompSize, stream);
    ASSERT(isPtrAligned<16>(preComp.get()));
    preCompForMatch(preComp.get(), desc, nbDesc, mUseSqrNormPreComp, mUseSift4b, mUseCascadeHash, mAccNbDesc, mDescAcc[0], mCasHashWeights[0], mCasHashBiasBase[0], mHostScratch.get(), mDeviceScratch.get(),
		[this, hImage](uint32_t maxBucketSize){
			{
				std::lock_guard lk{mLockForMaxBucketSize};
				ASSERT(mMaxBucketSizeMap.try_emplace(hImage, maxBucketSize).second);
			}
			mCVarForMaxBucketSize.notify_all();
		}, stream);
    const auto key = mBuilder.registerCacheableData<std_byte>(std::move(preComp), makeFmtStr("%u.matcherPreComp", static_cast<uint32_t>(hImage)), cudapp::storage::DiskStoragePolicy::kImmutable, false);
    {
        // std::lock_guard lk{mLock}; // already locked
        ASSERT(mPreCompDataKeys.try_emplace(hImage, key).second);
    }

    if (userStream != mStream.get()) {
        connectStreams(mStream.get(), userStream, mEvent.get(), &mEventLock);
    }
}

void BruteForceMatcher::matchImpl(const std::vector<Img>& queries, const Img& train, bool bidirectional)
{
    bidirectional = true; // For now we only instantiated bidirectional kernel for bruteforce matching.
    const uint32_t nbTasks = cast32u(queries.size());
    auto reserveStorage = [this, &queries, nbTasks](std::vector<MatchStorage>& storage, std::function<uint32_t(Img)> nbDescGetter){
        if (storage.size() < nbTasks) {
#if USE_DELEGATED_STREAM_SYNC
            cudapp::fiberSyncCudaStream(mStream.get());
#else
            cudaCheck(cudaStreamSynchronize(mStream.get()));
#endif
            storage.resize(nbTasks);
        }
        for (uint32_t i = 0; i < nbTasks; i++) {
            const auto& q = queries.at(i);
            auto& s = storage.at(i);
            const auto nbDesc = nbDescGetter(q);
            if (s.capacity < nbDesc) {
#if USE_DELEGATED_STREAM_SYNC
                cudapp::fiberSyncCudaStream(mStream.get());
#else
                cudaCheck(cudaStreamSynchronize(mStream.get()));
#endif
                s.devData = allocCudaMem<BestMatch, CudaMemType::kDevice>(nbDesc);
                s.hostData = allocCudaMem<BestMatch, CudaMemType::kPinned>(nbDesc);
                s.capacity = nbDesc;
            }
        }
    };
    reserveStorage(mMatchesFwd, [](const Img& q)->uint32_t{return q.nbDesc;});
    if (bidirectional) {
        reserveStorage(mMatchesBwd, [&train](const Img&)->uint32_t{return train.nbDesc;});
    }
    using cudapp::storage::StorageLocation;
    {
        std::vector<cudapp::storage::AcquiredMemory<const std_byte>> preCompHolders;
        preCompHolders.reserve(nbTasks * 2);
        if (mUseSift4b) {
            using Descriptor = Sift4bDescTraits::Descriptor;
            ASSERT(mUseSqrNormPreComp);
			ASSERT(!mUseCascadeHash);
            std::vector<Sift4bBruteForceMatchTask> cudaTasks(nbTasks);
			const auto& t = train;
			auto& manager = mBuilder.storageManager();
			auto preCompBHolder = cudapp::storage::acquireMemory<const std_byte>(manager, mPreCompDataKeys.at(t.hImage), StorageLocation::kCudaDeviceMem, mStream.get(), false, true);
            for (uint32_t i = 0; i < nbTasks; i++) {
                const auto& q = queries.at(i);
                assert(q.nbDesc <= mMatchesFwd.at(i).capacity);
                assert(!bidirectional || t.nbDesc <= mMatchesBwd.at(i).capacity);
                auto preCompAHolder = cudapp::storage::acquireMemory<const std_byte>(manager, mPreCompDataKeys.at(q.hImage), StorageLocation::kCudaDeviceMem, mStream.get(), false, true);
                cudaTasks.at(i) = Sift4bBruteForceMatchTask{
                    q.nbDesc, t.nbDesc,
                    reinterpret_cast<const Descriptor*>(preCompAHolder.data()),
                    reinterpret_cast<const Descriptor*>(preCompBHolder.data()),
                    mMatchesFwd.at(i).devData.get(),
                    mMatchesBwd.at(i).devData.get(),
                    reinterpret_cast<const uint32_t*>(preCompAHolder.data() + alignUp(sizeof(Descriptor) * q.nbDesc)),
                    reinterpret_cast<const uint32_t*>(preCompBHolder.data() + alignUp(sizeof(Descriptor) * t.nbDesc))
                };
                // Do not release before usage is dispatched to mStream.get()
                preCompHolders.emplace_back(std::move(preCompAHolder));
            }
			preCompHolders.emplace_back(std::move(preCompBHolder));
            launchBruteForceMatchIMMA(cudaTasks.data(), nbTasks, mStream.get());
        }
        else {
			const auto& t = train;
			auto& manager = mBuilder.storageManager();
			std::vector<const std_byte*> preCompDataA;
			std_byte const* preCompDataB = nullptr;
			if (mUseSqrNormPreComp || mUseCascadeHash) {
				for (uint32_t i = 0; i < nbTasks; i++) {
					const auto& q = queries.at(i);
					auto preCompAHolder = cudapp::storage::acquireMemory<const std_byte>(manager, mPreCompDataKeys.at(q.hImage), StorageLocation::kCudaDeviceMem, mStream.get(), false, true);
					preCompDataA.push_back(preCompAHolder.data());// Do not release before usage is dispatched to mStream.get()
                    preCompHolders.emplace_back(std::move(preCompAHolder));	
				}
			 	auto preCompBHolder = cudapp::storage::acquireMemory<const std_byte>(manager, mPreCompDataKeys.at(t.hImage), StorageLocation::kCudaDeviceMem, mStream.get(), false, true);
				preCompDataB = preCompBHolder.data();
				preCompHolders.emplace_back(std::move(preCompBHolder));
			}

			if (mUseCascadeHash) {
				std::vector<cashash::ImageDesc> casHashQueryImgs(nbTasks);
				std::vector<uint32_t> querySize(nbTasks);
				std::vector<uint32_t> maxQueryBucketSize(nbTasks);
				std::vector<cashash::MatchResult> results(nbTasks);
				auto makeImgDesc = [](const Img& img, const std_byte* preComp){
					const auto sqrNormDesc = reinterpret_cast<const uint32_t*>(preComp);
					preComp += alignUp(sizeof(uint32_t) * img.nbDesc);
					const auto bucketEnds = reinterpret_cast<const cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>*>(preComp);
					preComp += alignUp(sizeof(cudapp::KArray<uint32_t, cashash::nbScheme, cashash::nbBuckets>));
					const auto buckets = reinterpret_cast<const uint32_t*>(preComp);
					preComp += alignUp(sizeof(uint32_t) * img.nbDesc * cashash::nbScheme);
					return cashash::ImageDesc {
						img.desc,
						bucketEnds,
						buckets,
						sqrNormDesc
					};
				};
				auto getMaxBucketSize = [this](ImageHandle hImage){
					std::unique_lock lk{mLockForMaxBucketSize};
					mCVarForMaxBucketSize.wait(lk, [this, hImage](){
						return mMaxBucketSizeMap.find(hImage) != mMaxBucketSizeMap.end();
					});
					return mMaxBucketSizeMap.at(hImage);
				};
				for (uint32_t i = 0; i < nbTasks; i++) {
					const auto& q = queries.at(i);
					assert(q.nbDesc <= mMatchesFwd.at(i).capacity);
					assert(!bidirectional || t.nbDesc <= mMatchesBwd.at(i).capacity);
					casHashQueryImgs.at(i) = makeImgDesc(q, preCompDataA.at(i));
					querySize.at(i) = q.nbDesc;
					maxQueryBucketSize.at(i) = getMaxBucketSize(q.hImage);
					results.at(i) = {
						mMatchesFwd.at(i).devData.get(),
						mMatchesBwd.at(i).devData.get()
					};
				}
				const auto trainImg = makeImgDesc(t, preCompDataB);
				cashash::findBestMatch(casHashQueryImgs.data(), querySize.data(), trainImg, t.nbDesc, results.data(), nbTasks, maxQueryBucketSize.data(), getMaxBucketSize(t.hImage), mStream.get());
			}
            else {
            	std::vector<SiftBruteForceMatchTask> cudaTasks(nbTasks);
				for (uint32_t i = 0; i < nbTasks; i++) {
					const auto& q = queries.at(i);
					assert(q.nbDesc <= mMatchesFwd.at(i).capacity);
					assert(!bidirectional || t.nbDesc <= mMatchesBwd.at(i).capacity);
					cudaTasks.at(i) = SiftBruteForceMatchTask{
						q.nbDesc, t.nbDesc, q.desc, t.desc,
						mMatchesFwd.at(i).devData.get(),
						mMatchesBwd.at(i).devData.get(),
						nullptr, nullptr
					};
					if (mUseSqrNormPreComp || mUseCascadeHash) {
						cudaTasks.at(i).sqrNormDescA = reinterpret_cast<const uint32_t*>(preCompDataA.at(i));
						cudaTasks.at(i).sqrNormDescB = reinterpret_cast<const uint32_t*>(preCompDataB);			
					}
				}
				if (mUseSqrNormPreComp) {
					launchBruteForceMatchIMMA(cudaTasks.data(), nbTasks, mStream.get());
				}
				else {
					launchBruteForceMatchIDP(cudaTasks.data(), nbTasks, mStream.get());
				}
			}
        }
        // not needed but faster with stream sync here. no idea why. Maybe due to scheduling issues
#if USE_DELEGATED_STREAM_SYNC
        cudapp::fiberSyncCudaStream(mStream.get());
#else
        cudaCheck(cudaStreamSynchronize(mStream.get()));
#endif
    }
    for (uint32_t i = 0; i < nbTasks; i++) {
        const auto& q = queries.at(i);
		const auto& t = train;
        const auto& fwd = mMatchesFwd.at(i);
        const auto& bwd = mMatchesBwd.at(i);
        cudaCheck(cudaMemcpyAsync(fwd.hostData.get(), fwd.devData.get(), sizeof(BestMatch) * q.nbDesc, cudaMemcpyDeviceToHost, mStream.get()));
        if (bidirectional) {
            cudaCheck(cudaMemcpyAsync(bwd.hostData.get(), bwd.devData.get(), sizeof(BestMatch) * t.nbDesc, cudaMemcpyDeviceToHost, mStream.get()));
        }
    }
}

fb::future<std::vector<std::pair<std::vector<typename BruteForceMatcher::BestMatch>, std::vector<typename BruteForceMatcher::BestMatch>>>> BruteForceMatcher::match(
    const std::vector<Img>& queries, const Img& train, bool bidirectional, cudaStream_t userStream)
{
    std::lock_guard<fb::mutex> lk{mLock};
    fb::promise<std::vector<std::pair<std::vector<BestMatch>, std::vector<BestMatch>>>> resultPromise;
    auto resultFuture = resultPromise.get_future();
    if (queries.empty()) {
        resultPromise.set_value({});
        return resultFuture;
    }
    if (userStream != mStream.get()) {
        connectStreams(userStream, mStream.get(), mEvent.get(), &mEventLock);
    }
    matchImpl(queries, train, bidirectional);
    // copy to sys mem in mStream.get(), to avoid race between reserveStorage and host cross-check.
    launchCudaHostFunc(mStream.get(), [this, queries, train, bidirectional, resultPromise = std::move(resultPromise)]() mutable {
        std::vector<std::pair<std::vector<BestMatch>, std::vector<BestMatch>>> results(queries.size());
        for (size_t i = 0; i < queries.size(); i++) {
            results.at(i).first.assign(mMatchesFwd.at(i).hostData.get(), mMatchesFwd.at(i).hostData.get() + queries.at(i).nbDesc);
            if (bidirectional) {
                results.at(i).second.assign(mMatchesBwd.at(i).hostData.get(), mMatchesBwd.at(i).hostData.get() + train.nbDesc);
            }
        }
        resultPromise.set_value(std::move(results));
    });
    if (userStream != mStream.get()) {
        connectStreams(mStream.get(), userStream, mEvent.get(), &mEventLock);
    }
    return resultFuture;
}

fb::future<std::vector<std::vector<ValidMatch>>> BruteForceMatcher::matchAndFilter(
    const std::vector<Img>& queries, const Img& train, bool crossCheck, cudaStream_t userStream)
{
    std::lock_guard<fb::mutex> lk{mLock};
    fb::promise<std::vector<std::vector<ValidMatch>>> resultPromise;
    auto resultFuture = resultPromise.get_future();
    if (queries.empty()) {
        return resultFuture;
    }
    if (userStream != mStream.get()) {
        connectStreams(userStream, mStream.get(), mEvent.get(), &mEventLock);
    }
    matchImpl(queries, train, crossCheck);
    // cross-check in mStream.get(), to avoid race between reserveStorage and host cross-check.
    launchCudaHostFunc(mStream.get(), [this, queries, train, crossCheck, resultPromise = std::move(resultPromise)]() mutable {
        resultPromise.set_value(this->filterMatches(queries, train, crossCheck));
    });
    if (userStream != mStream.get()) {
        connectStreams(mStream.get(), userStream, mEvent.get(), &mEventLock);
    }
    return resultFuture;
}

std::vector<std::vector<ValidMatch>> BruteForceMatcher::filterMatches(const std::vector<Img>& queries, const Img& train, bool crossCheck) const
{
    std::vector<std::vector<ValidMatch>> results(queries.size());
    for (size_t n = 0; n < queries.size(); n++) {
        const auto nbDescA = queries.at(n).nbDesc;
        const auto nbDescB = train.nbDesc;
        const auto fwd = mMatchesFwd.at(n).hostData.get();
        const auto bwd = mMatchesBwd.at(n).hostData.get();
        auto& r = results.at(n);
        if (crossCheck) {
            r = crossCheckMatches(fwd, nbDescA, bwd, nbDescB);
        }
        else {
            r = removeMatchConflicts(fwd, nbDescA, nbDescB);
        }
    }
    return results;
}
} // namespace rsfm