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

#pragma once
#include "RapidSFM.h"
#include <vector>
#include <unordered_map>
#include "fwd.h"
#include <shared_mutex>
#include <cuda_utils.h>
#include <RapidBoW.h>
#include <fiberProxy.h>
#include <ArbitraryPool.h>
#include <Runtime.h>

namespace rsfm
{
class Builder : public IBuilder, public Runtime
{
public:
    Builder(std::unique_ptr<Config> cfg);
    ~Builder();

    PipelineType getPipelineType() const override { return PipelineType::kIncremental; };
    size_t getNbPipelineStages() const override;
    IntriType getIntriType() const override;
    size_t getObliqueCameraBundleSize() const override { return 1u; };

    void setProgressCallback(ProgressCallback callback, void* data) override;

    // called in main thread
    TiePtHandle addTiePoint() override;
    TiePtHandle addControlPoint(double x, double y, double z, const Covariance3& cov, float huber) override;
    ImageHandle addImage(const char* file, CameraHandle camera, PoseHandle pose,
        const TiePtMeasurement* tiePtMeasurements, size_t nbTiePtMeasurements) override;
    CameraHandle addCamera(uint32_t resX, uint32_t resY, float fx, float fy = kNaN, float cx = kNaN, float cy = kNaN, float k1 = 0.f, float k2 = 0.f, float p1 = 0.f, float p2 = 0.f, float k3 = 0.f) override;
    PoseHandle addPose(float rx = kNaN, float ry = kNaN, float rz = kNaN, double cx = kNaN, double cy = kNaN, double cz = kNaN, const Covariance3& cov = Covariance3::inf(), float huber = INFINITY, float vx = 0.f, float vy = 0.f, float vz = 0.f) override;

    void finish() override;

    size_t getNbModels() const override;
    const IModel* getModel(size_t idx) const override;

    void writePly(const std::string& filenamePattern = "cloud_%u.ply") const;
    void writeNvm(const std::string& filenamePattern = "cloud_%u.nvm") const;
    void writeRsm(const std::string& filenamePattern = "cloud_%u.rsm") const;

    void setSavedName(ImageHandle hImage, const char* name) override;
    void writeClouds(const char* filenamePrefix, uint32_t flag) const override;

public:
    std::vector<size_t> getPipeLineStatus() const;
    void getPipelineStatus(size_t* nbPendingTasksPerStage) const override;
    size_t getNbBlockingTasks() const;

public:
    cudapp::PriorityFiberPool* fiberPool() const {return mFiberPool.get();}
    cudapp::FiberBlockingService* fiberBlockingService() const {return mBlockingService.get();}
    const Config& config() const {return *mConfig;}
    Config& config() {return *mConfig;}

    RapidSift* siftDetector() const {return mSiftDetector.get();}
    const rbow::IVocabulary* vocabulary() const {return mVocabulary.get();}

    const LocCtrl* getCtrlPoint(TiePtHandle hCtrlPt) const;
    RealCamera* getRealCamera(CameraHandle hCamera) const;
    const Vec2<uint32_t>& getCamResolution(CameraHandle hCamera) const;
    Image* getImage(ImageHandle hImage) const;
	const Pose* getPose(PoseHandle hPose) const;
    const LocCtrl* getPoseGNSS(PoseHandle hPose) const;
    const char* getSavedName(ImageHandle hImage) const;

private:
    std::unique_ptr<cudapp::IPipeLine<std::tuple<fs::path, ImageHandle, CameraHandle, PoseHandle, std::vector<TiePtMeasurement>>, ImageHandle>> createPipeLine();

    void preprocessImage(fs::path file, ImageHandle hImage, CameraHandle hCamera, PoseHandle hPose, std::vector<TiePtMeasurement> tiePtMeasurement);
    std::vector<ImageHandle> searchAndAddToDataBase(ImageHandle hImage);
	std::vector<std::pair<ImageHandle, std::vector<std::pair<Index, Index>>>> findImagesWithCommonTiePts(ImageHandle hImage);
    std::vector<std::unique_ptr<Matches>> matchImages(
		std::vector<std::pair<ImageHandle, std::vector<std::pair<Index, Index>>>> hFirstCandidates, // hFirst and tie point measurement pairs
		ImageHandle hSecond);
    std::vector<std::unique_ptr<ImagePair>> solveImagePairs(ImageHandle hSecond, std::vector<std::unique_ptr<Matches>> allMatches);
    void addImageToModels(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> pairs);
    void bundleAdjustment();
    void concatDesc(ImageHandle hImage);
    void buildVocabulary(uint32_t nbImages);

private:
    void dbgDrawMatches(const Matches& matches, const char* prefix);
    void notifyProgress();
private:
    std::unique_ptr<Config> mConfig;

    mutable std::mutex mProgressCallbackLock;
    ProgressCallback mProgressCallback = {nullptr};
    void* mProgressCallbackData = nullptr;

    cudapp::ArbitraryPool<CudaStream> mRandStreams;
    CudaStream mImgSearchStream = makeCudaStreamWithPriority(-1);

    std::tuple< std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kDevice>>,
                std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kPinned>>,
                std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kSystem>>> mCudaMemPools;
    std::unique_ptr<cudapp::storage::StorageManager> mStorageManager;

    std::unique_ptr<RapidSift> mSiftDetector;
    std::unique_ptr<BruteForceMatcher> mMatcher;

    cudapp::ArbitraryPool<std::unique_ptr<RansacMatchFilter>> mMatchFilters;

    std::unique_ptr<HandleGenerator<TiePtHandle>> mTiePtHandleGenerator;
    std::unique_ptr<HandleGenerator<CameraHandle>> mCameraHandleGenerator;
    std::unique_ptr<HandleGenerator<PoseHandle>> mPoseHandleGenerator;
    std::unique_ptr<HandleGenerator<ImageHandle>> mImageHandleGenerator;

    // protects mCameras, mPose, mImages, mCamImages, mPoseImages and mImageSavedNames but not the data pointed to by std::unique_ptr.
    // unique_ptr makes it safe to access the data wihout being impacted by container operation
	// if value is std::vector instead of unique_ptr, the content is protected by this lock.
    mutable std::shared_mutex mLock;
    struct TiePtRef {ImageHandle hImage; uint32_t idxTiePtInImg;};
    std::unordered_map<TiePtHandle, std::vector<TiePtRef>, DirectMappingHash<TiePtHandle>> mTiePoints; // lock, get ptr+size then unlock & return
    std::unordered_map<TiePtHandle, std::unique_ptr<const LocCtrl>, DirectMappingHash<TiePtHandle>> mCtrlPoints;
    std::unordered_map<CameraHandle, std::unique_ptr<RealCamera>, DirectMappingHash<CameraHandle>> mRealCameras;
    std::unordered_map<CameraHandle, std::unique_ptr<const Vec2<uint32_t>>, DirectMappingHash<CameraHandle>> mCamResolutions;
    std::unordered_map<PoseHandle, std::unique_ptr<Pose>, DirectMappingHash<PoseHandle>> mPoses;
    std::unordered_map<PoseHandle, std::unique_ptr<const LocCtrl>, DirectMappingHash<PoseHandle>> mPoseGNSS;
    std::unordered_map<ImageHandle, std::unique_ptr<Image>, DirectMappingHash<ImageHandle>> mImages;
    std::unordered_map<CameraHandle, std::vector<ImageHandle>, DirectMappingHash<CameraHandle>> mCamImages; // not used anywhere, yet
    std::unordered_map<PoseHandle, std::vector<ImageHandle>, DirectMappingHash<PoseHandle>> mPoseImages; // not used anywhere, yet
    std::unordered_map<ImageHandle, std::string> mImageSavedNames; // names to use when saving NVM/RSM files

    // for vocabulary creation
    std::vector<std::array<uint8_t, 128>> mAllDesc;

    // immutable vocabulary after creation.
    std::unique_ptr<rbow::IVocabulary> mVocabulary;
    // used only in the main loop fiber
    std::unique_ptr<rbow::IDataBase> mDatabase;

    std::unique_ptr<ModelBuilder> mModelBuilder;

    std::unique_ptr<cudapp::FiberBlockingService> mBlockingService; // One thread for light-weight blocking tasks, e.g. disk/network IO and cuda sync, etc.
    std::unique_ptr<cudapp::PriorityFiberPool> mFiberPool; // Pool for CPU-intensive tasks

    std::unique_ptr<cudapp::ConcurrentQueue<ImageHandle, fb::mutex, fb::condition_variable_any>> mFinishedImages;
    //fixme: don't forget to set capacity of the first input channel to a large value
    std::unique_ptr<cudapp::IPipeLine<std::tuple<fs::path, ImageHandle, CameraHandle, PoseHandle, std::vector<TiePtMeasurement>>, ImageHandle>> mPipeLine;
};

} // namespace rsfm
