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
#include "DefaultModel.h"
#include <VectorMap.h>
#include "../HandleGenerator.h"
#include <unordered_map>
#include <vector>
#include "../RapidSFM.h"
#include "../fwd.h"
#include "../rsm.h"
#include "DebugModelViewer.h"
#include "../Types.hpp"
#include <unordered_set>

namespace rsfm{

struct Point
{
    Vec3<float> location;
    float avgSqrErr; // use padding space to store
    std::vector<DefaultModel::IdxOb> observations;
};

inline const Point nullPoint{Vec3f{NAN, NAN, NAN}, INFINITY, {}};

class IncreModel : public DefaultModel
{
public:
    IncreModel(Builder& builder);
    ~IncreModel();
    void collectUnusedPairs(std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>& unusedPairs);
    float checkAffinity(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs);
    bool addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>>& allPairs) override;
    bool isInitialized() const {return !mObPoints.empty();}
    enum BAMethod {
        kGlobal,
        kLocal,
        kSemiGlobal // iterative and like local but recursively update those impacted by the last iteration
    };
    void bundleAdjustment(BAMethod method, bool isFinal);
    void finish() override {
        retriangulate();
        bundleAdjustment(BAMethod::kGlobal, true);
        mPtColor = computePointColor();
    }
    std::vector<std::pair<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>>
        takeAllPendingPairs();
    std::vector<std::unique_ptr<ImagePair>> takePendingPairs(ImageHandle hImage);
    // merge src into this model
    bool tryMergeModel(IncreModel& src, std::vector<std::unique_ptr<ImagePair>>& pairs);
    void writePly(const char* filename) const final;
    void writeNvm(const char* filename) const final;
    void writeRsm(const char* filename) const final;
    const std::vector<ImageHandle>& getPending() const {return mPendingSeq;}
    const std::vector<ImageHandle>& getFusedImages() const {return mFusedImgSeq;}
    bool hasFused(ImageHandle hImage) const {return mObPoints.find(hImage) != mObPoints.end();}
    size_t getNbPoints() const {return mPoints.size();}
    void printSummary(std::ostream& o);
    bool isPending(ImageHandle hImage) const;
    void showModel();
	void rescaleModel(float scale);
	void transform(const Sim3Transform& sim3);
    void retriangulate();
private:
    rsm::RapidSparseModel getModel() const;
    void serialize(std::ostream& stream) const;
    bool tryInitializeModelImpl();
    // pairs may be updated by removing used pairs
    // @fixme: we don't need ImagePair. rsfm::Match should be sufficient after initialization
    bool tryAddImageImpl(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>>& pairs);

	bool checkIsHardCtrlPt(PointHandle hPoint) const;
	bool checkIsHardCtrlPt(TiePtHandle hTiePt) const;

    struct CacheEntry;
    using ImgInfoCache = std::unordered_map<ImageHandle, std::unique_ptr<CacheEntry>>;
    const CacheEntry& cachedFind(ImageHandle hImage, ImgInfoCache& cache) const;

    void retriangulatePair(ImgInfoCache& cache, ImagePair const& pair, int32_t idxSolution = -1);

    void mergePoints(PointHandle dst, PointHandle src, ImgInfoCache& cache);

    // return flag indicating if the point location is changed.
    // trianguate with the last ob combined with each of the other ob, then find the new location with minimum avgSqrErr for reprojection.
    bool updatePointWithLastOb(PointHandle hPoint, ImgInfoCache& cache);
    void sanifyPointObs(PointHandle hPoint, ImgInfoCache& cache, float sqrErrThres);
    void sanifyLastOb(PointHandle hPoint, ImgInfoCache& cache, float sqrErrThres);

    void addNewImgCovisibility(ImageHandle hImage, const ImageHandle* neighbour, size_t nbNeighbours);

    void sanityCheck();

    VectorMap<PointHandle, Color> computePointColor() const; // for all points

    void localBA(ImgInfoCache& cache);
    void globalBA(ImgInfoCache& cache, bool isFinal);
    void bundleAdjustmentImpl(const std::vector<ImageHandle>& varImages, float huber, bool optimizeIntrinsics, bool isFinal, ImgInfoCache& cache);
    // src2dst is the merge links
    void mergeModelImpl(IncreModel& src, const Sim3Transform& transform, 
        const std::unordered_multimap<PointHandle, PointHandle, DirectMappingHash<PointHandle>>& src2dst,
        const std::vector<std::pair<ImageHandle, ImageHandle>>& pairs);

	bool tryAlign();
	bool isAligned() const {return mIsAligned;}
	void sanifyRollingShutterVelocity();
	std::vector<std::pair<PointHandle, const LocCtrl*>> getCtrlPts() const;
	uint32_t getNbCtrlPts() const;
	std::vector<std::pair<PoseHandle, const LocCtrl*>> getPoseGnss() const;

    // for re-triangulation
    int32_t getMatchingSolution(ImagePair const& p) const;
    // -1 means auto-detect with getMatchingSolution(p)
    float getConstructedMatchRatio(ImagePair const& p, int32_t idxSolution = -1) const;
    bool isWellConstructed(ImagePair const& p) const;
private:
    Builder* mBuilder;
    // Not empty only when waiting for initialization
    std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>, DirectMappingHash<ImageHandle>> mPendingPairs;
    std::vector<ImageHandle> mPendingSeq;

    // inverse pairs with hSecond not added to this model but hFirst is.
    std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>, DirectMappingHash<ImageHandle>> mUnusedPairs;

    HandleGenerator<PointHandle> mPointHandleGenerator;

    VectorMap<PointHandle, Point> mPoints;
    VectorMap<PoseHandle, Pose> mPoses;
    VectorMap<CameraHandle, RealCamera> mCameras;
    VectorMap<PointHandle, Color> mPtColor;

    std::unordered_map<ImageHandle, VectorMap<IdxKPt, PointHandle>, DirectMappingHash<ImageHandle>> mObPoints;
    std::vector<ImageHandle> mFusedImgSeq;
    VectorMap<ImageHandle, std::vector<ImageHandle>> mCovisibility;

    std::vector<std::unique_ptr<ImagePair>> mUnderConstructedPairs;
    std::vector<std::unique_ptr<ImagePair>> mWellConstructedPairs;

	std::unordered_map<PointHandle, TiePtHandle, DirectMappingHash<PointHandle>> mTiePts;
	std::unordered_set<PoseHandle, DirectMappingHash<PoseHandle>> mGnssPose; // only include pose of fused captures

	bool mIsAligned = false;

    std::vector<ImageHandle> mPendingLocalBA;
    size_t mLastGlobalBaNbImages = 0;
    std::unique_ptr<rba::IUniversalModel> mBundleModel;

    std::unique_ptr<DebugModelViewer> mViewer;    
};

} // namespace rsfm

