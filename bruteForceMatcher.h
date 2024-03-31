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

#include "bruteForceMatch.h"
#include <vector>
#include <cpp_utils.h>
#include "fwd.h"
#include "RapidSFM.h"
#include <fiberProxy.h>

namespace rsfm
{

class BruteForceMatcher
{
public:
    using Descriptor = typename SiftDescTraits::Descriptor;
    BruteForceMatcher(Builder& builder);
    BruteForceMatcher(const BruteForceMatcher&) = delete;
    BruteForceMatcher& operator=(const BruteForceMatcher&) = delete;
    ~BruteForceMatcher();
    //! Full content of desc may NOT be stored, so you still need to provide when you call match()
    //! The intention here is to do pre-compute, and transform 8-bit to 4-bit SIFT if required.
    void registerImage(ImageHandle hImage, const Descriptor* desc, uint32_t nbDesc, cudaStream_t stream);

    using Distance = typename SiftDescTraits::Distance;
    using BestMatch = BestMatchImpl<Distance>;
    struct MatchTask
    {
        const Descriptor* descA;
        uint32_t nbDescA;
        ImageHandle hImageA;
        const Descriptor* descB;
        uint32_t nbDescB;
        ImageHandle hImageB;
    };
	struct Img{
        const Descriptor* desc;
        uint32_t nbDesc;
        ImageHandle hImage;
	};
    fb::future<std::vector<std::vector<ValidMatch>>> matchAndFilter(
        const std::vector<Img>& queries, const Img& train, bool crossCheck, cudaStream_t stream);
    fb::future<std::vector<std::pair<std::vector<BestMatch>, std::vector<BestMatch>>>> match(
        const std::vector<Img>& queries, const Img& train, bool bidirectional, cudaStream_t stream);
private:
    // dispatch matching kernel and copy to MatchStorage::hostData. Work is done in mStream
    void matchImpl(const std::vector<Img>& queries, const Img& train, bool bidirectional);
    // to be dispatched to mStream.get()
    std::vector<std::vector<ValidMatch>> filterMatches(const std::vector<Img>& queries, const Img& train, bool crossCheck) const;
    bool needPreCompStorage() const;
private:
    Builder& mBuilder;
    const bool mUseSqrNormPreComp;
    const bool mUseSift4b;
	const bool mUseCascadeHash;

    mutable fb::mutex mLock;

    using CacheObjKeyType = cudapp::storage::CacheObjKeyType;
    std::unordered_map<ImageHandle, CacheObjKeyType> mPreCompDataKeys;

	// for cascade hashing
	uint64_t mAccNbDesc = 0;
	CudaMem<cudapp::KArray<uint64_t, 128>, CudaMemType::kDevice> mDescAcc;

	mutable fb::mutex mLockForMaxBucketSize;
	std::unordered_map<ImageHandle, uint32_t> mMaxBucketSizeMap; // for cascade hashing
	fb::condition_variable mCVarForMaxBucketSize;

	CudaMem<cudapp::KArray<int32_t, 64, 32>, CudaMemType::kDevice> mCasHashWeights; // int8_t[64][128]
	CudaMem<cudapp::KArray<int32_t, 64>, CudaMemType::kDevice> mCasHashBiasBase;
	CudaMem<std_byte, CudaMemType::kPinned> mHostScratch; // for maxBucketSize precompute
	CudaMem<std_byte, CudaMemType::kDevice> mDeviceScratch; // for cascade hash precompute
	size_t mDeviceScratchSize{0};

    struct MatchStorage
    {
        uint32_t capacity;
        CudaMem<BestMatch, CudaMemType::kDevice> devData;
        CudaMem<BestMatch, CudaMemType::kPinned> hostData;
    };
    std::vector<MatchStorage> mMatchesFwd;
    std::vector<MatchStorage> mMatchesBwd;
    CudaStream mStream = makeCudaStream();
    CudaEvent mEvent = makeCudaEvent();
    mutable std::mutex mEventLock; // locked when connectStreams(), to make connectStreams() atomic.
};

} // namespace rsfm
