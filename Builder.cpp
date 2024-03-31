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

#include <DefaultCacheableObject.h>
#include <CudaMemPool.h>
#include "Image.h"
#include <FiberUtils.h>
#include "HandleGenerator.h"
#include "Types.h"
#include <RapidSift.h>
#include <boost/fiber/cuda/waitfor.hpp>
#include <algorithm>
#include "ImagePair.h"
#include "Types.h"
#include <PipeLine.h>
#include "Builder.h"
#include <Runtime.hpp>
#include "Config.h"
#include <macros.h>
#include "models/ModelBuilder.h"
#include <Profiler.h>
#include "bruteForceMatcher.h"
#include <PriorityFiberPool.h>
#include "Scheduler.h"
#include "RansacMatchFilter.h"
#include <unordered_set>

using cudapp::storage::StorageLocation;
using cudapp::storage::DefaultCacheableObject;
using cudapp::storage::StorageManager;

#define DRAW_MATCHES 0
#define DRAW_PAIRS 0

namespace
{
std::unique_ptr<rbow::IVocabulary> tryLoadVocabulary(const fs::path& filename) {
    if (!fs::exists(filename)) {
        return nullptr;
    }
    const auto blob = loadBinaryFile(filename);
    return rbow::deserializeVocabulary(reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
}
}

namespace rsfm
{
Builder::~Builder() {
    for (auto& item : mImages) {
        auto& img = item.second;
        storageManager().removeItem(img->keyPoints);
        img->keyPoints = cudapp::storage::kInvalidKey;
        storageManager().removeItem(img->descriptors);
        img->descriptors = cudapp::storage::kInvalidKey;
        storageManager().removeItem(img->kptsColor);
        img->kptsColor = cudapp::storage::kInvalidKey;
    }
}

Builder::Builder(std::unique_ptr<Config> cfg)
    : Runtime(cfg->cacheFolder.c_str(), cfg->nbRandStreams,
        cfg->memPool.deviceBytes, cfg->memPool.pinnedBytes, cfg->memPool.sysBytes,
        cfg->objCache.deviceBytes, cfg->objCache.pinnedBytes, cfg->objCache.sysBytes)
    , mConfig{cfg == nullptr ? std::make_unique<Config>() : std::move(cfg)}
    , mRandStreams{config().nbRandStreams, makeCudaStream, cudaStreamDefault}
    , mCudaMemPools{
          std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kDevice>>(config().memPool.deviceBytes),
          std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kPinned>>(config().memPool.pinnedBytes),
          std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kSystem>>(config().memPool.sysBytes)}
    , mStorageManager{std::make_unique<StorageManager>(config().objCache.deviceBytes, config().objCache.pinnedBytes, config().objCache.sysBytes)}
    , mSiftDetector{create_sift(config().sift.nbWorkers, config().sift.descType)}
    , mMatcher{std::make_unique<BruteForceMatcher>(*this)}
    , mMatchFilters{config().matchFilter.nbWorkers, [this](){return std::make_unique<RansacMatchFilter>(*this);}}
	, mTiePtHandleGenerator{std::make_unique<HandleGenerator<TiePtHandle>>()}
    , mCameraHandleGenerator{std::make_unique<HandleGenerator<CameraHandle>>()}
    , mPoseHandleGenerator{std::make_unique<HandleGenerator<PoseHandle>>()}
    , mImageHandleGenerator{std::make_unique<HandleGenerator<ImageHandle>>()}
    , mVocabulary(config().vocabulary.rebuild ? nullptr : tryLoadVocabulary(config().getVocabularyPath()))
    , mDatabase(mVocabulary == nullptr ? nullptr : rbow::createDataBase(mVocabulary.get()))
    , mModelBuilder{std::make_unique<ModelBuilder>(*this)}
    , mBlockingService { std::make_unique<cudapp::FiberBlockingService>(std::chrono::milliseconds(config().blockingService.interval), config().blockingService.windowSize, config().blockingService.maxPendingTasks) }
    , mFiberPool { std::make_unique<cudapp::PriorityFiberPool>(config().fiberPool.nbThreads, config().fiberPool.stackSize) }
    , mFinishedImages{ std::make_unique<cudapp::ConcurrentQueue<ImageHandle, fb::mutex, fb::condition_variable_any> >(std::numeric_limits<uint32_t>::max()) }
    , mPipeLine{createPipeLine()}
{
    if (!fs::exists(config().cacheFolder)) {
        fs::create_directories(config().cacheFolder);
    }
}

size_t Builder::getNbPipelineStages() const {return mPipeLine->getNbStages();}

IntriType Builder::getIntriType() const { return config().opticModel; }

template <typename PipeLine>
std::vector<size_t> getStatusOfPipeLine(const PipeLine& pipe) {
    std::vector<size_t> status(pipe.getNbStages() + 1);
    for (unsigned i = 0; i < pipe.getNbStages() + 1; i++) {
        status[i] = pipe.getCurrentChannelSize(i);
    }
    return status;
}

void Builder::getPipelineStatus(size_t* nbPendingTasksPerStage) const {
    const auto tmp = getPipeLineStatus();
    assert(tmp.size() == getNbPipelineStages() + 1);
    std::copy(tmp.begin(), tmp.end(), nbPendingTasksPerStage);
}

std::vector<size_t> Builder::getPipeLineStatus() const
{
    return getStatusOfPipeLine(*mPipeLine);
}

size_t Builder::getNbBlockingTasks() const
{
    return fiberBlockingService()->getNbTasks();
}

TiePtHandle Builder::addControlPoint(double x, double y, double z, const Covariance3& cov, float huber)
{
    // RapidBA does not allow that the dimensions of a control point to be mixed hard/soft
    ASSERT((cov.xx == 0) == (cov.yy == 0) && (cov.yy == 0) == (cov.zz == 0));
    const TiePtHandle hCtrlPt = addTiePoint();
    {
        std::lock_guard<std::shared_mutex> lk{mLock};
        mCtrlPoints.try_emplace(hCtrlPt, std::make_unique<LocCtrl>(LocCtrl{
            {static_cast<Coord>(x), static_cast<Coord>(y), static_cast<Coord>(z)},
            huber, cov}));
    }
    return hCtrlPt;
}

void Builder::setProgressCallback(ProgressCallback callback, void* data) {
    const std::lock_guard<std::mutex> lk{mProgressCallbackLock};
    mProgressCallback = callback;
    mProgressCallbackData = data;
}

TiePtHandle Builder::addTiePoint()
{
    const TiePtHandle hTiePt = mTiePtHandleGenerator->make();
    {
        std::lock_guard<std::shared_mutex> lk{mLock};
        mTiePoints.try_emplace(hTiePt, std::vector<TiePtRef>{});
    }
    return hTiePt;
}

ImageHandle Builder::addImage(const char* file, CameraHandle hCamera, PoseHandle hPose,
    const TiePtMeasurement* tiePtMeasurements, const size_t nbTiePtMeasurements)
{
    // hImage must be generated outside of fiber pool to guarantee order, and also for return value.
    const ImageHandle hImage = mImageHandleGenerator->make();
    {
        std::shared_lock<std::shared_mutex> lk{mLock};
        ASSERT(hPose == kInvalid<PoseHandle> || mPoses.find(hPose) != mPoses.end());
        ASSERT(mRealCameras.find(hCamera) != mRealCameras.end());
    }

    if (hPose == kInvalid<PoseHandle>)
    {
        hPose = addPose(kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, Covariance3::inf(), INFINITY);
    }

	std::vector<TiePtMeasurement> tmp{tiePtMeasurements, tiePtMeasurements + nbTiePtMeasurements};
	std::sort(tmp.begin(), tmp.end(), MemberLess<TiePtMeasurement, TiePtHandle, &TiePtMeasurement::hTiePt>{});
	if (std::adjacent_find(tmp.begin(), tmp.end(), MemberEqual<TiePtMeasurement, TiePtHandle, &TiePtMeasurement::hTiePt>{}) != tmp.end()) {
		throw std::runtime_error("Multiple measurements of the same tie point");
	}
	if (nbTiePtMeasurements != 0)
    {
        std::lock_guard<std::shared_mutex> lk{mLock};
        mCamImages.at(hCamera).push_back(hImage);
        mPoseImages.at(hPose).push_back(hImage);
        for (uint32_t i = 0; i < cast32u(nbTiePtMeasurements); i++) {
            const auto& m = tmp.at(i);
            mTiePoints.at(m.hTiePt).push_back(TiePtRef{hImage, i});
        }
    }
    mPipeLine->enqueue(std::make_tuple(fs::path{file}, hImage, hCamera, hPose, std::move(tmp)));

    return hImage;
}

CameraHandle Builder::addCamera(uint32_t resX, uint32_t resY, float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2, float k3)
{
    ASSERT(std::isfinite(fx) && std::isfinite(fy) && std::isfinite(cx));
    switch (getIntriType()) {
        case IntriType::kF1:
        case IntriType::kF1D5:
        case IntriType::kF1C2D5:
        case IntriType::kF1D2: ASSERT(fx == fy && "fx == fy is required"); break;
        default: break;
    }
    std::lock_guard<std::shared_mutex> lk{mLock};
    const CameraHandle hCamera = mCameraHandleGenerator->make();
    const Vec2<uint32_t> resolution = {resX, resY};
    ASSERT(mCamResolutions.try_emplace(hCamera, std::make_unique<Vec2<uint32_t>>(resolution)).second);
    const RealCamera realCam{{{fx, fy},
        {std::isfinite(cx) ? cx : float(resX / 2), std::isfinite(cy) ? cy : float(resY / 2)}},
        {{k1, k2, p1, p2, k3}}};
    ASSERT(mRealCameras.try_emplace(hCamera, std::make_unique<RealCamera>(realCam)).second);
    ASSERT(mCamImages.try_emplace(hCamera).second);
    return hCamera;
}

PoseHandle Builder::addPose(float rx, float ry, float rz, double cx, double cy, double cz, const Covariance3& cov, float huber, float vx, float vy, float vz)
{
	{
		if (!cov.isInf()) {
			ASSERT(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cz));
		}
		const Vec3f v{vx, vy, vz};
		if (config().shutterType == ShutterType::kRolling1D) {
			ASSERT(std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz));
			ASSERT(v.squaredNorm() != 0);
		}
		if (config().shutterType == ShutterType::kGlobal) {
			ASSERT(v.squaredNorm() == 0);
		}
	}
    std::lock_guard<std::shared_mutex> lk{mLock};
    const PoseHandle hPose = mPoseHandleGenerator->make();
    std::unique_ptr<Pose> pose;
    const bool isValidC = std::isfinite(cx) && std::isfinite(cy) && std::isfinite(cz);
	const bool isValidR = std::isfinite(rx) && std::isfinite(ry) && std::isfinite(rz);
	pose = std::make_unique<Pose>(Pose{
		isValidR ? rvec2quat({rx, ry, rz}) : Rotation{kNaN, kNaN, kNaN, kNaN},
		isValidC ? Vec3f{static_cast<Coord>(cx), static_cast<Coord>(cy), static_cast<Coord>(cz)} : Vec3f{kNaN, kNaN, kNaN},
		{vx, vy, vz}
	});
    ASSERT(mPoses.try_emplace(hPose, std::move(pose)).second);
    if (!cov.isInf()) {
        mPoseGNSS.try_emplace(hPose, std::make_unique<LocCtrl>(LocCtrl{
            {static_cast<Coord>(cx), static_cast<Coord>(cy), static_cast<Coord>(cz)},
            huber, cov
        }));
    }
    ASSERT(mPoseImages.try_emplace(hPose).second);
    return hPose;
}

const LocCtrl* Builder::getCtrlPoint(TiePtHandle hCtrlPt) const
{
    std::shared_lock<std::shared_mutex> lk{mLock};
    const auto iter = mCtrlPoints.find(hCtrlPt);
    if (iter == mCtrlPoints.end()) {
        return nullptr;
    }
    return iter->second.get();
}

const Pose* Builder::getPose(PoseHandle hPose) const {
    std::shared_lock<std::shared_mutex> lk{mLock};
	const auto iter = mPoses.find(hPose);
	if (iter == mPoses.end()) {
		return nullptr;
	}
	return iter->second.get();
}

const LocCtrl* Builder::getPoseGNSS(PoseHandle hPose) const {
    std::shared_lock<std::shared_mutex> lk{mLock};
    const auto iter = mPoseGNSS.find(hPose);
    if (iter == mPoseGNSS.end()) {
        return nullptr;
    }
    return iter->second.get();

}

RealCamera *Builder::getRealCamera(CameraHandle hCamera) const{
    std::shared_lock<std::shared_mutex> lk{mLock};
    return mRealCameras.at(hCamera).get();
}

const Vec2<uint32_t>& Builder::getCamResolution(CameraHandle hCamera) const {
    std::shared_lock<std::shared_mutex> lk{mLock};
    return *mCamResolutions.at(hCamera);
}

Image* Builder::getImage(ImageHandle hImage) const{
    std::shared_lock<std::shared_mutex> lk{mLock};
    return mImages.at(hImage).get();
}

const char* Builder::getSavedName(ImageHandle hImage) const {
    std::shared_lock<std::shared_mutex> lk{mLock};
    const auto iter = mImageSavedNames.find(hImage);
    return iter == mImageSavedNames.end() ? mImages.at(hImage)->file.c_str() : iter->second.c_str();
}

// PinHoleCamera *Builder::getCamera(CameraHandle hCamera) const{
//     std::shared_lock<std::shared_mutex> lk{mLock};
//     return mCameras.at(hCamera).get();
// }

std::unique_ptr<cudapp::IPipeLine<std::tuple<fs::path, ImageHandle, CameraHandle, PoseHandle, std::vector<TiePtMeasurement>>, ImageHandle>> Builder::createPipeLine()
{
    auto stagePreprocessImage = [this](std::tuple<fs::path, ImageHandle, CameraHandle, PoseHandle, std::vector<TiePtMeasurement>> in)
            -> fb::future<ImageHandle>
    {
        auto asyncTask = [this, in{std::move(in)}]{
            preprocessImage(std::move(std::get<fs::path>(in)), std::get<ImageHandle>(in), std::get<CameraHandle>(in), std::get<PoseHandle>(in), std::move(std::get<std::vector<TiePtMeasurement>>(in)));
            return std::get<ImageHandle>(in);
        };
        return fiberPool()->post(std::move(asyncTask));
    };
    // optional stage
    auto stageConcatDesc = [this](fb::future<ImageHandle> f) -> ImageHandle{
        const auto hImage = f.get();
        concatDesc(hImage);
        return hImage;
    };
    // optional stage
    auto stageBuildVoc = [this](cudapp::PipeLineChannel<ImageHandle>& in, cudapp::PipeLineChannel<fb::future<ImageHandle>>& out) {
        in.waitClosed();
        buildVocabulary(cast32u(in.peekSize()));

        while (true) {
            auto h = in.pop(); // in is closed, so will not wait even if empty
            if (!h.has_value()) {
                break;
            }
            fb::promise<ImageHandle> p{};
            out.emplace(p.get_future());
            p.set_value(h.value());
        }
    };
	// returns candidate handles and common tie points
    auto stageFindPairingCandidates = [this](fb::future<ImageHandle> futureImage)
            -> std::pair<std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>>, ImageHandle>
    {
        const ImageHandle hImg = futureImage.get();
		const auto tiedCandidates = findImagesWithCommonTiePts(hImg);
        const auto nbNeighbours = std::min(config().pair.nbNeighbours, static_cast<uint32_t>(hImg));
        const auto searchedCandidates = searchAndAddToDataBase(hImg);

		std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>> candidates;
		candidates.reserve(tiedCandidates.size() + nbNeighbours + searchedCandidates.size());
		candidates = tiedCandidates;

		static const auto toCandNoTiePt = [](ImageHandle h) {return std::pair<ImageHandle, std::vector<Pair<Index>>>{h, {}};};

		std::generate_n(std::back_inserter(candidates), nbNeighbours, [x = static_cast<uint32_t>(hImg)]()mutable{
			return toCandNoTiePt(static_cast<ImageHandle>(--x));
		});
		std::transform(searchedCandidates.begin(), searchedCandidates.end(), std::back_inserter(candidates), toCandNoTiePt);

		std::unordered_set<ImageHandle> dedup;
		candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const auto& cand){return !dedup.insert(cand.first).second;}), candidates.end());

        if (candidates.size() > config().pair.maxNbCandidates) {
            candidates.resize(config().pair.maxNbCandidates);
        }
		candidates.shrink_to_fit();

        return std::make_pair(std::move(candidates), hImg);
    };
    auto stageMatchImages = [this](std::pair<std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>>, ImageHandle> pairCandidates)
            -> std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<Matches>>>>
    {
        const ImageHandle hSecond = pairCandidates.second;
        auto asyncTask = [this, pairCandidates{std::move(pairCandidates)}]{
            return matchImages(std::move(pairCandidates.first), pairCandidates.second);
        };
        return std::make_pair(hSecond, fiberPool()->post(std::move(asyncTask)));
    };
    auto stageDrawMatch = [this](std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<Matches>>>> in)
            -> std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<Matches>>>>
    {
        std::vector<std::unique_ptr<Matches>> allMatches = in.second.get();
        std::vector<fb::future<void> > fibers;
        auto asyncTask = [this](Matches* matches){
            dbgDrawMatches(*matches, "match");
        };
        for (const auto& matches : allMatches) {
            fibers.emplace_back(fiberPool()->post(asyncTask, matches.get()));
        }
        return std::make_pair(in.first, fiberPool()->post(
            [allMatches{std::move(allMatches)}, fibers{std::move(fibers)}]() mutable -> std::vector<std::unique_ptr<Matches>>{
                for (auto& f : fibers) { f.get(); }
                return std::move(allMatches);
            }));
    };
    unused(stageDrawMatch);
    auto stageSolvePairs = [this](std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<Matches>>>> in)
        -> std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>>
    {
        auto asyncTask = [this, hSecond{in.first}, allMatches{in.second.get()}]() mutable ->std::vector<std::unique_ptr<ImagePair>>{
            return solveImagePairs(hSecond, std::move(allMatches));
        };
        return std::make_pair(in.first, fiberPool()->post(std::move(asyncTask)));
    };
    auto stageDrawPair = [this](std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>> in)
            -> std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>>
    {
        std::vector<std::unique_ptr<ImagePair>> allPairs = in.second.get();
        std::vector<fb::future<void> > fibers;
        auto asyncTask = [this](ImagePair* imgPair){
            Matches matches {imgPair->images, {}, imgPair->tiePtMatches};
            const auto idxInliers = imgPair->solutions.at(0).inliers;
            std::transform(idxInliers.begin(), idxInliers.end(), std::back_inserter(matches.kptsMatches), [imgPair](uint32_t idx){
                return imgPair->kptsMatches.at(idx);
            });
            dbgDrawMatches(matches, "pair");
        };
        for (const auto& imgPair : allPairs) {
            fibers.emplace_back(fiberPool()->post(asyncTask, imgPair.get()));
        }
        return std::make_pair(in.first, fiberPool()->post(
            [allPairs{std::move(allPairs)}, fibers{std::move(fibers)}]()mutable
            -> std::vector<std::unique_ptr<ImagePair>> {
                for (auto& f : fibers) { f.get(); }
                return std::move(allPairs);
            }));
    };
    unused(stageDrawPair);
    // optional scheduler
    auto stageSchedule = [this](
        cudapp::PipeLineChannel<std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>>>& in,
        cudapp::PipeLineChannel<std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>>>& out)
    {
        in.waitClosed();
        auto trace = cudapp::Profiler::instance().mark("scheduler");
        const auto nbInputs = in.peekSize();
        Scheduler sched;
        std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>> allPairs;
        while (true) {
            auto input = in.pop();
            if (!input.has_value()) {
                break;
            }
            auto [hImage, futPairs] = std::move(input.value());
            auto pairs = futPairs.get();
            sched.add(hImage, pairs);
            ASSERT(allPairs.emplace(hImage, std::move(pairs)).second);
        }
        const auto seq = sched.getSequence();
        trace.reset();
        ASSERT(nbInputs == seq.size());
        for (const ImageHandle h : seq) {
            fb::promise<std::vector<std::unique_ptr<ImagePair>>> p;
            out.emplace(h, p.get_future());
            p.set_value(std::move(allPairs.at(h)));
            allPairs.erase(h);
        }
    };
    auto stageUpdateModel = [this](std::pair<ImageHandle, fb::future<std::vector<std::unique_ptr<ImagePair>>>> in)
            -> ImageHandle
    {
        mModelBuilder->addImage(in.first, in.second.get());
        return in.first;
    };
    auto stageTerminal = [](ImageHandle in)
            -> ImageHandle
    {
        return in;
    };
#if DRAW_PAIRS || DRAW_MATCHES
    printf("debugging with pair drawing\n");
#endif

    if (mVocabulary == nullptr) {
        printf("Vocabulary file not found. We will build one.\n");
        if (config().model.useScheduler){
            return cudapp::makePipeLine([this](int32_t priority, std::function<void()> f){return fiberPool()->post(priority, f);}, *mFinishedImages,
                                        std::move(stagePreprocessImage),
                                        std::move(stageConcatDesc),
                                        std::move(stageBuildVoc),
                                        std::move(stageFindPairingCandidates),
                                        std::move(stageMatchImages),
#if DRAW_MATCHES
                                        std::move(stageDrawMatch),
#endif
                                        std::move(stageSolvePairs),
#if DRAW_PAIRS
                                        std::move(stageDrawPair),
#endif
                                        std::move(stageSchedule),
                                        std::move(stageUpdateModel),
                                        std::move(stageTerminal));
        }
        else {
            return cudapp::makePipeLine([this](int32_t priority, std::function<void()> f){return fiberPool()->post(priority, f);}, *mFinishedImages,
                                        std::move(stagePreprocessImage),
                                        std::move(stageConcatDesc),
                                        std::move(stageBuildVoc),
                                        std::move(stageFindPairingCandidates),
                                        std::move(stageMatchImages),
#if DRAW_MATCHES
                                        std::move(stageDrawMatch),
#endif
                                        std::move(stageSolvePairs),
#if DRAW_PAIRS
                                        std::move(stageDrawPair),
#endif
                                        std::move(stageUpdateModel),
                                        std::move(stageTerminal));
        }
    }
    else {
        if (config().model.useScheduler){
            return cudapp::makePipeLine([this](int32_t priority, std::function<void()> f){return fiberPool()->post(priority, f);}, *mFinishedImages,
                                        std::move(stagePreprocessImage),
                                        std::move(stageFindPairingCandidates),
                                        std::move(stageMatchImages),
#if DRAW_MATCHES
                                        std::move(stageDrawMatch),
#endif
                                        std::move(stageSolvePairs),
#if DRAW_PAIRS
                                        std::move(stageDrawPair),
#endif
                                        std::move(stageSchedule),
                                        std::move(stageUpdateModel),
                                        std::move(stageTerminal));
        }
        else {
            return cudapp::makePipeLine([this](int32_t priority, std::function<void()> f){return fiberPool()->post(priority, f);}, *mFinishedImages,
                                        std::move(stagePreprocessImage),
                                        std::move(stageFindPairingCandidates),
                                        std::move(stageMatchImages),
#if DRAW_MATCHES
                                        std::move(stageDrawMatch),
#endif
                                        std::move(stageSolvePairs),
#if DRAW_PAIRS
                                        std::move(stageDrawPair),
#endif
                                        std::move(stageUpdateModel),
                                        std::move(stageTerminal));
        }
    }
}

void Builder::preprocessImage(fs::path file, ImageHandle hImage, CameraHandle hCamera, PoseHandle hPose, std::vector<TiePtMeasurement> tiePtMeasurements)
{
    const Image image = createImage(this, file, hImage, hCamera, hPose, std::move(tiePtMeasurements), siftDetector());
    // register in matcher
    {
        const cudaStream_t stream = anyStream();
        constexpr auto kCudaDeviceMem = cudapp::storage::StorageLocation::kCudaDeviceMem;
        const auto descHolder = cudapp::storage::acquireMemory<const SiftDescriptor>(storageManager(), image.descriptors, kCudaDeviceMem, stream, false, true);
        static_assert(sizeof(SiftDescriptor) == sizeof(rsfm::BruteForceMatcher::Descriptor));
        mMatcher->registerImage(image.hImage, reinterpret_cast<const rsfm::BruteForceMatcher::Descriptor*>(descHolder.data()), descHolder.nbElems(), stream);
    }
    {
        std::lock_guard<std::shared_mutex> lk{mLock};
        const auto emplaceResult = mImages.try_emplace(hImage, std::make_unique<Image>(image));
        ASSERT(emplaceResult.second);
    }
    notifyProgress();
}

cudaError_t cudaPickAbstractDesc(SiftDescriptor* abstractDesc, const SiftDescriptor* descriptors,
    const uint32_t* sampleIndices, uint32_t nbSamples, cudaStream_t stream);

std::vector<ImageHandle> Builder::searchAndAddToDataBase(ImageHandle hImage)
{
    const auto trace = cudapp::Profiler::instance().mark(__func__);
    const cudaStream_t stream = mImgSearchStream.get();
    const Image& img = [this, hImage]()->const Image&{
        std::shared_lock<std::shared_mutex> lk{mLock};
        return *mImages.at(hImage).get();
    }();

    const size_t nbAbstractSamples = img.abstract.size();
    const auto abstractDesc = cudaMemPool<CudaMemType::kDevice>().alloc<SiftDescriptor>(nbAbstractSamples, stream);
    {
        auto pinnedAbstractIndices = cudaMemPool<CudaMemType::kPinned>().alloc<uint32_t>(nbAbstractSamples, stream);
        launchCudaHostFunc(stream,
        [abstract{std::move(img.abstract)}, pinnedAbstractIndexPtr{pinnedAbstractIndices.get()}]{
            std::copy(abstract.begin(), abstract.end(), pinnedAbstractIndexPtr);
        });
        img.abstract = {};
        auto devAbstractIndices = cudaMemPool<CudaMemType::kDevice>().alloc<uint32_t>(nbAbstractSamples, stream);
        // unfortunately, stream priority does not work for memcpy
        cudaCheck(cudaMemcpyAsync(devAbstractIndices.get(), pinnedAbstractIndices.get(),
            sizeof(uint32_t) * nbAbstractSamples, cudaMemcpyHostToDevice, stream));

        constexpr auto kCudaDeviceMem = cudapp::storage::StorageLocation::kCudaDeviceMem;
        const auto descHolder = cudapp::storage::acquireMemory<const SiftDescriptor>(storageManager(), img.descriptors, kCudaDeviceMem, stream, false, true);
        assert(descHolder.loc() == kCudaDeviceMem);
        assert(img.nbKPoints == descHolder.nbElems());
        cudaCheck(cudaPickAbstractDesc(abstractDesc.get(), descHolder.data(),
            devAbstractIndices.get(), nbAbstractSamples, stream));
    }

    const auto indicesInLeafLevel = cudaMemPool<CudaMemType::kDevice>().alloc<uint32_t>(nbAbstractSamples, stream);
    const auto hostIndicesInLeafLevel = cudaMemPool<CudaMemType::kPinned>().alloc<uint32_t>(nbAbstractSamples, stream);

    vocabulary()->lookUp(abstractDesc.get(), nbAbstractSamples, indicesInLeafLevel.get(), stream);
    cudaCheck(cudaMemcpyAsync(hostIndicesInLeafLevel.get(), indicesInLeafLevel.get(), sizeof(uint32_t) * nbAbstractSamples, cudaMemcpyDeviceToHost, stream));

    cudapp::fiberSyncCudaStream(stream);
    static_assert (std::is_same_v<std::underlying_type_t<ImageHandle>, uint32_t>);
    const auto candidateIndices = mDatabase->queryAndAddDoc(static_cast<uint32_t>(hImage), hostIndicesInLeafLevel.get(), nbAbstractSamples, config().pair.maxNbCandidates);
    std::vector<ImageHandle> candidates(candidateIndices.size());
    std::transform(candidateIndices.begin(), candidateIndices.end(), candidates.begin(), [](uint32_t idx){return static_cast<ImageHandle>(idx);});
    notifyProgress();
    return candidates;
}

std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>> Builder::findImagesWithCommonTiePts(ImageHandle hImage) {
	const Image& img = [this, hImage]()->const Image&{
        std::shared_lock<std::shared_mutex> lk{mLock};
        return *mImages.at(hImage).get();
    }();
	std::unordered_map<ImageHandle, std::vector<Pair<Index>>> candidates;
	{
		std::shared_lock<std::shared_mutex> lk{mLock};
		for (uint32_t i = 0; i < img.getNbTiePtMeasurements(); i++) {
			const auto& tiePtRefs = mTiePoints.at(img.tiePtMeasurements[i].hTiePt);
			for (const auto& r : tiePtRefs) {
				if (r.hImage < hImage) {
					candidates[r.hImage].emplace_back(r.idxTiePtInImg, i);
				}
			}
		}
	}
	std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>> results{candidates.begin(), candidates.end()};
	std::sort(results.begin(), results.end(), [](const auto& x, const auto& y){return x.second.size() > y.second.size();});
	return results;
}

size_t Builder::getNbModels() const {
    return mModelBuilder->getNbModels();
}

const IModel* Builder::getModel(size_t idx) const {
    return toInterface(mModelBuilder->getModel(idx));
}

void Builder::finish() {
    mPipeLine->close();
    mModelBuilder->finish();
}

void Builder::writePly(const std::string& filenamePattern) const
{
    mModelBuilder->writePly(filenamePattern);
}
void Builder::writeNvm(const std::string& filenamePattern) const
{
    mModelBuilder->writeNvm(filenamePattern);
}
void Builder::writeRsm(const std::string& filenamePattern) const
{
    mModelBuilder->writeRsm(filenamePattern);
}

void Builder::setSavedName(ImageHandle hImage, const char* name) {
    std::lock_guard<std::shared_mutex> lk{mLock};
    mImageSavedNames[hImage] = name;
}

void Builder::writeClouds(const char* filenamePrefix, uint32_t flag) const {
    std::vector<fb::future<void>> futures;
    futures.reserve(3);
    if (flag & kPLY) {
        futures.emplace_back(fiberPool()->post([&](){writePly(std::string{filenamePrefix} + "%u.ply");}));
    }
    if (flag & kRSM) {
        futures.emplace_back(fiberPool()->post([&](){writeRsm(std::string{filenamePrefix} + "%u.rsm");}));
    }
    if (flag & kNVM) {
        futures.emplace_back(fiberPool()->post([&](){writeNvm(std::string{filenamePrefix} + "%u.nvm");}));
    }
    for (auto& f : futures) {
        f.get();
    }
}

}// namespace rsfm
