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

// #pragma GCC optimize(0)

#include "IncreModel.h"
#include <RapidBA.h>
#include "IncreModel.hpp"
#include "../Config.h"
#include <unordered_map>
#include "../Builder.h"
#include "../Image.h"
#include "rbaUtils.h"
#include "../Types.hpp"
#include "../legacy/geometry.hpp"
#include "../ransac.hpp"
#include <Profiler.h>
#include <random>

namespace rsfm
{

void IncreModel::localBA(ImgInfoCache& cache) {
    if (mPendingLocalBA.empty()){
        return;
    }

    std::unordered_set<ImageHandle> varImages;
    for (const ImageHandle primary : mPendingLocalBA) {
        varImages.insert(primary);
        for (const ImageHandle secondary : mCovisibility.at(primary)) {       
            varImages.insert(secondary);
        }
    }
    bundleAdjustmentImpl({varImages.begin(), varImages.end()}, mBuilder->config().bundle.huberLocal, false, false, cache);
    mPendingLocalBA.clear();
}

void IncreModel::globalBA(ImgInfoCache& cache, bool isFinal) {
    if (mObPoints.empty()) {
        return;
    }

    std::vector<ImageHandle> varImages;
    varImages.reserve(mObPoints.size());
    for (const auto& item : mObPoints) {
        varImages.emplace_back(item.first);
    }
    const auto& cfg = mBuilder->config();
    bool const optimizeIntri = cfg.bundle.optimizeIntrinsics && (isFinal || varImages.size() >= cfg.bundle.minNbCapturesForIntrinsicsOptimization);
    bundleAdjustmentImpl({varImages.begin(), varImages.end()}, cfg.bundle.huberGlobal, optimizeIntri, isFinal, cache);
    mPendingLocalBA.clear();
}

#pragma GCC diagnostic push
// #pragma GCC optimize ("O0")
void IncreModel::bundleAdjustmentImpl(const std::vector<ImageHandle>& varImages, float huber, bool optimizeIntrinsics, bool isFinal, ImgInfoCache& cache)
{
    const auto trace = cudapp::Profiler::instance().mark(__func__); //__PRETTY_FUNCTION__
    if (varImages.empty()){
        return;
    }

	const auto& cfg = mBuilder->config();
    std::unordered_set<ImageHandle> varImgSet(varImages.begin(), varImages.end());
	const auto shutter = (!cfg.rollingShutterOnlyInFinalBA || isFinal) ? cfg.shutterType : ShutterType::kGlobal;
	const auto requiredShutter = (!isAligned() && (shutter == ShutterType::kRollingFixedVelocity || (shutter == ShutterType::kRolling1D))) ? ShutterType::kGlobal : shutter;
	// printf("shutter = %u, requiredShutter = %u\n", (uint32_t)shutter, (uint32_t)requiredShutter);
    if (mBundleModel == nullptr || mBundleModel->getIntriType() != toRBA(mBuilder->config().opticModel) || mBundleModel->getShutterType() != requiredShutter/* || (mBundleModel->getMayShareIntrinsics() != mBuilder->config().bundle.useGrpModel)*/) { //@fixme
        mBundleModel.reset(rba::createUniversalModel(mBuilder->config().bundle.useGrpModel, toRBA(mBuilder->config().opticModel), requiredShutter));
    }
    else {
        mBundleModel->clear();
    }
    std::unordered_map<ImageHandle, rba::IdxCap, DirectMappingHash<ImageHandle>> varCaps;
    std::unordered_map<ImageHandle, rba::IdxCap, DirectMappingHash<ImageHandle>> fixedCaps;
    std::unordered_map<PointHandle, rba::IdxPt, DirectMappingHash<PointHandle>> varPts; // also include soft control points
    std::unordered_map<CameraHandle, rba::IdxCam, DirectMappingHash<CameraHandle>> cameras;
    std::unordered_map<PointHandle, rba::IdxPt, DirectMappingHash<PointHandle>> hardCtrlPts; // hard control points only. Soft control points are in varPts and softCtrlPtGrdResAcc
    std::unordered_map<PoseHandle, std::unique_ptr<std::pair<float, uint32_t>>, DirectMappingHash<PoseHandle>> poseGrdResAcc; // ground resolution accumulator for omega of pose GNSS.h
	std::unordered_map<PointHandle, std::unique_ptr<std::pair<float, uint32_t>>> softCtrlPtGrdResAcc;

    auto tryAddCap = [this, &cache, &varCaps, &fixedCaps, &cameras, &poseGrdResAcc](ImageHandle hImage, bool fixed) {
        if (varCaps.count(hImage) != 0 || fixedCaps.count(hImage) != 0) {
            return kInvalid<rba::IdxCap>;
        }
        const auto& entry = cachedFind(hImage, cache);
        if (cameras.count(entry.hCamera) == 0){
			const uint32_t camResY = cast32u(entry.image->height);
			assert(camResY == mBuilder->getCamResolution(entry.hCamera).y);
            const rba::UniversalIntrinsics intri = toRBA(mCameras.at(entry.hCamera), camResY);
            const rba::IdxCam idxCam = mBundleModel->addCamera(intri, true);
            ASSERT(cameras.try_emplace(entry.hCamera, idxCam).second);
        }
		Pose& pose = mPoses.at(entry.hPose);
		const auto pGnss = mBuilder->getPoseGNSS(entry.hPose);
		// if it's fixed loc, set it when !fixed so BA will optimize it.
		if (isAligned() && !fixed && pGnss != nullptr && pGnss->cov.isFixed()) {
			pose.C = pGnss->loc;
		}
        const rba::IdxCap idxCap = mBundleModel->addCapture(cameras.at(entry.hCamera), toRBA(pose), fixed);
		if (!fixed && pGnss != nullptr) {
			if (poseGrdResAcc.count(entry.hPose) == 0) {
				ASSERT(poseGrdResAcc.try_emplace(entry.hPose, std::make_unique<std::pair<float, uint32_t>>(0.f, 0u)).second);
			}
		}
        return idxCap;
    };
    const float avgKPtSizeSqr = square(cfg.sift.upSample ? 2.25f : 5.f);
    const auto getWeight = [avgKPtSizeSqr](half size){
        return avgKPtSizeSqr / square(__half2float(size));
    };
    for (const ImageHandle hImage : varImages) {
        const rba::IdxCap idxCap = tryAddCap(hImage, false);
        ASSERT(idxCap != kInvalid<rba::IdxCap>);
        varCaps.try_emplace(hImage, idxCap);
        const auto& obPts = mObPoints.at(hImage);
        const auto& entry = cachedFind(hImage, cache);
		const KeyPoint* kpts = entry.keyPoints.data();
        const auto pt2dGetter = entry.makePt2dGetter();
        const auto pose = mPoses.at(entry.hPose);
        const auto opticAxis = pose.R.conjugate() * Vec3f{0,0,1};
        const auto getPtDepth = [&pose, &opticAxis](const Vec3f& pt)->float{ return (pt - pose.C).dot(opticAxis); };
        const bool needDepthAcc = isAligned() && (poseGrdResAcc.find(entry.hPose) != poseGrdResAcc.end());
        const auto pDepthAcc = needDepthAcc ? poseGrdResAcc.at(entry.hPose).get() : nullptr;
        const float fRcp = [&](){const auto f = mCameras.at(entry.hCamera).pinHole.f; return 1.f / sqrt(f.x * f.y);}();
        const auto accPosePtDepth = [&getPtDepth, &pDepthAcc, &fRcp](const Vec3f& pt){
            pDepthAcc->first += square(getPtDepth(pt) * fRcp);
            pDepthAcc->second++;
        };
        for (auto iter = obPts.begin(); iter != obPts.end(); iter++) {
            const IdxKPt idxKPt = iter.key();
            const PointHandle hPoint = *iter;
			const bool isTiePt = entry.isTiePt(idxKPt);
			const TiePtHandle hTiePt = isTiePt ? mTiePts.at(hPoint) : kInvalid<TiePtHandle>;
			assert(!isTiePt || entry.image->tiePtMeasurements.at(idxKPt - entry.image->nbKPoints).hTiePt == hTiePt);
			const LocCtrl* const pPtLocCtrl = isTiePt ? mBuilder->getCtrlPoint(hTiePt) : nullptr;
			const bool isCtrlPt = isTiePt && pPtLocCtrl != nullptr;
			const bool isHardCtrlPt = isCtrlPt && pPtLocCtrl->cov.isFixed();
			if (isAligned() && isHardCtrlPt) [[unlikely]] {
				assert(isCtrlPt && isHardCtrlPt);
				assert(mPoints.at(hPoint).location == pPtLocCtrl->loc);
				if (hardCtrlPts.count(hPoint) == 0) {
					const Point& p = mPoints.at(hPoint);
					const bool fixed = isAligned();
					const rba::IdxPt idxPt = mBundleModel->addPoint(toRBA(p.location), fixed);
					ASSERT(hardCtrlPts.try_emplace(hPoint, idxPt).second);
				}
			}
			else [[likely]] {
				if (varPts.count(hPoint) == 0) {
					const Point& p = mPoints.at(hPoint);
					const rba::IdxPt idxPt = mBundleModel->addPoint(toRBA(p.location), false);
					ASSERT(varPts.try_emplace(hPoint, idxPt).second);
					if (isAligned() && isCtrlPt)
					{
						assert(!isHardCtrlPt);
						auto iterAcc = softCtrlPtGrdResAcc.find(hPoint);
						if (iterAcc == softCtrlPtGrdResAcc.end()) {
							iterAcc = softCtrlPtGrdResAcc.try_emplace(hPoint, std::make_unique<std::pair<float, uint32_t>>(0.f, 0u)).first;
						}
						iterAcc->second->second += square(getPtDepth(pPtLocCtrl->loc) * fRcp);
						iterAcc->second->second++;
					}
				}
			}
            const rba::IdxPt idxPt = isAligned() && isHardCtrlPt ? hardCtrlPts.at(hPoint) : varPts.at(hPoint);
			if (isTiePt) [[unlikely]] {
				mBundleModel->addObservation(idxCap, idxPt, pt2dGetter(idxKPt), cfg.bundle.tiePtWeight, INFINITY);
			}
			else [[likely]] {
            	mBundleModel->addObservation(idxCap, idxPt, kpts[idxKPt].location, getWeight(kpts[idxKPt].size), huber);
			}
            if (needDepthAcc) {
                accPosePtDepth(mPoints.at(hPoint).location);
            }
        }
    }
	// Now accumulation for ground resolution of soft control points is done.
	// We are ready to set soft control points
	if (isAligned()) {
		for (const auto& [hPoint, pAcc] : softCtrlPtGrdResAcc) {
			const TiePtHandle hTiePt = mTiePts.at(hPoint);
			const LocCtrl* const pPtLocCtrl = mBuilder->getCtrlPoint(hTiePt);
			assert(pPtLocCtrl != nullptr);
			float omega[3][3];
			toInfoMat(omega, pPtLocCtrl->cov, pAcc->second != 0 ? std::sqrt(pAcc->first / pAcc->second) : 0.025f);
			mBundleModel->setSoftCtrlPoint(varPts.at(hPoint), toRBA(pPtLocCtrl->loc), omega, pPtLocCtrl->huber);
		}
	}
	// Some points are observed by fixed captures. Add these observations.
    std::unordered_set<PointHandle, DirectMappingHash<PointHandle>> ptsWithFixedCapObs;
    for (const auto& [hPoint, idxPt] : varPts) {
        const auto& obs = mPoints.at(hPoint).observations;
        for (const auto idxOb : obs) {
            const Observation ob = decodeIdxOb(idxOb);
            if (varImgSet.count(ob.hImage) != 0){
                continue;
            }
            ptsWithFixedCapObs.emplace(hPoint);
            if (fixedCaps.count(ob.hImage) == 0) {
                const rba::IdxCap idxCap = tryAddCap(ob.hImage, true);
                ASSERT(fixedCaps.try_emplace(ob.hImage, idxCap).second);
            }
			const auto& entry = cachedFind(ob.hImage, cache);
			if (entry.isTiePt(ob.idxKPt)) [[unlikely]] {
				mBundleModel->addObservation(fixedCaps.at(ob.hImage), idxPt, entry.getPt2d(ob.idxKPt), cfg.bundle.tiePtWeight, INFINITY);
			}
			else [[likely]] {
            	const auto& kpt = entry.keyPoints.data()[ob.idxKPt];
				mBundleModel->addObservation(fixedCaps.at(ob.hImage), idxPt, kpt.location, getWeight(kpt.size), huber);
			}
        }
    }
	size_t nbVarPoseGnss = 0;
	if (isAligned()) {
		for (const auto& [hImage, idxCap] : varCaps) {
			const auto& entry = cachedFind(hImage, cache);
			const auto pGnss = mBuilder->getPoseGNSS(entry.hPose);
			if (pGnss) {
				float omega[3][3];
				float huber;
				if (pGnss->cov.isFixed()) {
					std::fill_n(&omega[0][0], 9, INFINITY);
					huber = INFINITY;
				}
				else {
					const auto& acc = *poseGrdResAcc.at(entry.hPose);
					toInfoMat(omega, pGnss->cov, acc.second != 0 ? std::sqrt(acc.first / acc.second) : 0.025f);
					huber = pGnss->huber;
				}
				mBundleModel->setCaptureGNSS(idxCap, toRBA(pGnss->loc), omega, huber);
				nbVarPoseGnss++;
			}
		}
		ASSERT(nbVarPoseGnss == poseGrdResAcc.size());
	}
    //@fixme: maybe move this logic to RapidBA.
    if (fixedCaps.empty() && hardCtrlPts.empty() && nbVarPoseGnss == 0 && mBundleModel->isGrouped()){
        const ImageHandle fixedImg = varImages.at(0);
        const rba::IdxCap fixedCap = varCaps.at(fixedImg);
        mBundleModel->setCaptureFixed(fixedCap, true);
        ASSERT(fixedCaps.try_emplace(fixedImg, fixedCap).second);
        varCaps.erase(fixedImg);
    }

    // @fixme: change to detecting if all images of one camera is included in BA
    if (optimizeIntrinsics && mBundleModel->isGrouped()) {
        ASSERT(varCaps.size() + 1 >= mFusedImgSeq.size() && "Need to detect if all images of the cameras are includes, otherwise it may go wrong.");
        for (const auto& item : cameras) {
            mBundleModel->setCameraFixed(item.second, false);
        }
    }

    mBundleModel->setVerbose(false);
    mBundleModel->filterModel();
    {
        const auto mark = cudapp::Profiler::instance().mark("BA init");
        mBundleModel->initializeOptimization();
    }
    mBundleModel->optimize(60);
	for (const auto& item : varCaps) {
		const PoseHandle hPose = cachedFind(item.first, cache).hPose;
		auto& pose = mPoses.at(hPose);
		pose = fromRBA(mBundleModel->getCaptureParams(item.second));
		if ((isAligned() && isRolling(cfg.shutterType)) || cfg.shutterType == ShutterType::kRolling1DLoc) {
			const auto& ref = mBuilder->getPose(hPose)->v;
			switch (cfg.shutterType) {
			case ShutterType::kGlobal: assert(pose.v.squaredNorm() == 0); break;
			case ShutterType::kRolling1D:
			case ShutterType::kRolling1DLoc:
			{
				const auto v = ref * (pose.v.dot(ref) / ref.squaredNorm());
				pose.v = v;
				break;
			}
			case ShutterType::kRolling3D: break;
			case ShutterType::kRollingFixedVelocity: assert(pose.v == ref); break;
			}
		}
#if 0
		const Vec3f v = requiredShutter == ShutterType::kRolling1DLoc ? pose.v : pose.R * -pose.v; // always print in T-fashion
		printf("hPose= %u, r = {%f, %f, %f, %f}, c = {%f, %f, %f}, v = {%f, %f, %f} * 1E-4\n", (uint32_t)hPose, pose.R.w, pose.R.x, pose.R.y, pose.R.z, pose.C.x, pose.C.y, pose.C.z, v.x * 1E4f, v.y * 1E4f, v.z * 1E4f);
#endif
	}
    if (optimizeIntrinsics) {
        for (const auto& item : cameras) {
            const auto cam = mBundleModel->getCameraParams(item.second);
			assert(std::isnan(cam.rollingCenter) || cam.rollingCenter == mBuilder->getCamResolution(item.first).y * 0.5f);
            mCameras.at(item.first) = fromRBA(cam);
            // printf("f = {%f, %f}, c = {%f, %f}, d = {%f, %f, %f, %f, %f}\n", cam.f.x, cam.f.y, cam.c.x, cam.c.y, cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
            // printf("{fx,fy,cx,cy,k1,k2,p1,p2,k3} = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", cam.f.x, cam.f.y, cam.c.x, cam.c.y, cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
        }
    }
	// printf("Before BA: %lu poses, %lu points, %lu observations\n", mPoses.size(), mPoints.size(), std::accumulate(mObPoints.begin(), mObPoints.end(), size_t{0}, [](size_t acc, const std::pair<const ImageHandle, VectorMap<IdxKPt, PointHandle>>& x){return acc + x.second.size();}));
    for (const auto& item : varPts) {
        mPoints.at(item.first).location = fromRBA(mBundleModel->getPointPosition(item.second));
        sanifyPointObs(item.first, cache, square(ptsWithFixedCapObs.count(item.first) == 0 ? huber * 4 : huber * 8));
    }
	// printf("After BA: %lu poses, %lu points, %lu observations, huber = %f\n", mPoses.size(), mPoints.size(), std::accumulate(mObPoints.begin(), mObPoints.end(), size_t{0}, [](size_t acc, const std::pair<const ImageHandle, VectorMap<IdxKPt, PointHandle>>& x){return acc + x.second.size();}), huber);
}
#pragma GCC diagnostic pop

void IncreModel::bundleAdjustment(BAMethod method, bool isFinal){
    ImgInfoCache cache;
    switch (method)
    {
    case BAMethod::kLocal: ASSERT(!isFinal); localBA(cache); return;
    case BAMethod::kGlobal: globalBA(cache, isFinal); return;
    case BAMethod::kSemiGlobal: DIE("not implemented");
        break;
    }
    DIE("You should never reach here");
}

namespace
{
Sim3Transform findSim3Ransac(const std::array<Eigen::ArrayXf, 3>& src, const std::array<Eigen::ArrayXf, 3>& dst, float threshold, float confidence, size_t maxNbTests)
{
    printf("****************************************\n");
    auto ransacTest = [&src, &dst, &threshold](const std::array<uint32_t, 3>& indices) -> size_t
    {
        const auto srcSeed = sampleWithIndices(src, indices);
        const auto dstSeed = sampleWithIndices(dst, indices);
        const Sim3Transform sim3 = findSim3(srcSeed, dstSeed);
		const Eigen::Matrix<float, 3, 4> t = toAffine3f(sim3).matrix().topRows<3>();
        const auto seedErr = (srcSeed.rowwise().homogeneous() * t.transpose() - dstSeed).rowwise().squaredNorm().eval();
        if (!(seedErr.array() < square(threshold)).all()) {
            static int counter = 0; // not thread-safe but we don't care
            if (counter < 8) {
                printf("wrong samples: %f, %f, %f\n", std::sqrt(seedErr[0]), std::sqrt(seedErr[1]), std::sqrt(seedErr[2]));
                counter++;
            }
            return 0u;
        }
        const auto nbInliers = 
            ((t(0, 0) * src[0] + t(0, 1) * src[1] + t(0, 2) * src[2] + t(0, 3) - dst[0]).square() +
            (t(1, 0) * src[0] + t(1, 1) * src[1] + t(1, 2) * src[2] + t(1, 3) - dst[1]).square() +
            (t(2, 0) * src[0] + t(2, 1) * src[1] + t(2, 2) * src[2] + t(2, 3) - dst[2]).square()
            < square(threshold)).count();
        assert(nbInliers == checkSim3(t, src, dst, threshold).count());
        // printf("ransac sim3: %ld/%ld\n", nbInliers, src.at(0).rows());
        return static_cast<size_t>(nbInliers);
    };
    const std::array<uint32_t, 3> bestIndices = ransac<decltype(ransacTest), 3>(ransacTest, src[0].rows(), confidence, maxNbTests);
    return findSim3(sampleWithIndices(src, bestIndices), sampleWithIndices(dst, bestIndices));
}
} // unnamed namespace

#pragma GCC diagnostic push
// #pragma GCC optimize ("O0")
bool IncreModel::tryMergeModel(IncreModel& src, std::vector<std::unique_ptr<ImagePair>>& pairs)
{
    IncreModel& dst = *this;
    ASSERT(src.isInitialized() && dst.isInitialized());
    ASSERT(src.mBuilder == dst.mBuilder);
    ASSERT(src.mPoints.size() <= dst.mPoints.size());
	//@fixme: PG-18 optional global BA for both model here
    ImgInfoCache srcCache, dstCache;
    for (const auto& p : pairs){
        const ImageHandle hFirst = p->images.first;
        const ImageHandle hSecond = p->images.second;
        ASSERT(src.hasFused(hFirst));
        ASSERT(dst.hasFused(hSecond));
        src.mPendingLocalBA.push_back(hFirst);
        dst.mPendingLocalBA.push_back(hSecond);
    }
    std::sort(src.mPendingLocalBA.begin(), src.mPendingLocalBA.end());
    src.mPendingLocalBA.erase(std::unique(src.mPendingLocalBA.begin(), src.mPendingLocalBA.end()), src.mPendingLocalBA.end());
    src.retriangulate(); // @fixme: only triangulate those involving images in mPendingLocalBA.
    src.localBA(srcCache);
    std::sort(dst.mPendingLocalBA.begin(), dst.mPendingLocalBA.end());
    dst.mPendingLocalBA.erase(std::unique(dst.mPendingLocalBA.begin(), dst.mPendingLocalBA.end()), dst.mPendingLocalBA.end());
    dst.retriangulate(); // @fixme: only triangulate those involving images in mPendingLocalBA.
    dst.localBA(dstCache);

    std::unordered_set<PointHandle> srcPts;
    std::unordered_multimap<PointHandle, PointHandle, DirectMappingHash<PointHandle>> src2dst;
    for (const auto& p : pairs){
        const ImageHandle hFirst = p->images.first;
        const ImageHandle hSecond = p->images.second;
        ASSERT(src.hasFused(hFirst));
        ASSERT(dst.hasFused(hSecond));
        const auto& srcObPts = src.mObPoints.at(hFirst);
        const auto& dstObPts = dst.mObPoints.at(hSecond);
        for (const auto& m : p->kptsMatches){
            if (srcObPts.has(m.first) && dstObPts.has(m.second)) {
                src2dst.emplace(srcObPts.at(m.first), dstObPts.at(m.second));
                srcPts.emplace(srcObPts.at(m.first));
            }
        }
    }
    const auto& cfg = mBuilder->config().model.merge;
    const auto minNbCommonPoints = pairs.size() < cfg.nbLinksForForcedMerge ? cfg.minNbCommonPoints : cfg.forcedMergeMinNbCommonPoints;
    if (srcPts.size() < minNbCommonPoints) {
        return false;
    }
    // remove duplicate dst points for each src point
    {
        std::unordered_set<PointHandle, DirectMappingHash<PointHandle>> dstFound;
        for (PointHandle hSrc : srcPts) {
            const auto range = src2dst.equal_range(hSrc);
            for (auto iter = range.first; iter != range.second;) {
                if (dstFound.count(iter->second) == 0) {
                    dstFound.emplace(iter->second);
                    iter++;
                }
                else {
                    const auto victim = iter;
                    iter++;;
                    src2dst.erase(victim);
                }
            }
            dstFound.clear();
        }
    }
    std::array<Eigen::ArrayXf, 3> ptPairs[2];
    for (auto& x : ptPairs) {
        for (auto& y : x) {
            y.resize(src2dst.size());
        }
    }
    {
        uint32_t idxRow = 0;
        for (const auto [hSrc, hDst] : src2dst) {
            auto fillRow = [](std::array<Eigen::ArrayXf, 3>& target, uint32_t idxRow, const Vec3f& loc) {
                target[0][idxRow] = loc.x;
                target[1][idxRow] = loc.y;
                target[2][idxRow] = loc.z;
            };
            fillRow(ptPairs[0], idxRow, src.mPoints.at(hSrc).location);
            fillRow(ptPairs[1], idxRow, mPoints.at(hDst).location);
            idxRow++;
        }
    }
    
    const float refDepth = [&](){
        const auto ref = pairs.at(0)->images.second;
        const Eigen::Isometry3f& p = toEigen(Transform::fromPose(mPoses.at(mBuilder->getImage(ref)->hPose)));
        float acc = 0;
        int counter = 0;
        for (const auto& hPoint : mObPoints.at(ref)) {
            acc += (p * toEigen(mPoints.at(hPoint).location))[2];
            counter++;
        }
        return acc / counter;
    }();
    const Sim3Transform sim3trans = findSim3Ransac(ptPairs[0], ptPairs[1], refDepth * cfg.ransacThreshold, cfg.ransacConfidence, cfg.maxNbRansacTests);
    if (checkSim3(toAffine3f(sim3trans).matrix().topRows<3>(), ptPairs[0], ptPairs[1], refDepth * cfg.ransacThreshold).count()
        < std::max(minNbCommonPoints, static_cast<uint32_t>(src2dst.size() * cfg.minInlierRatio))) {
        return false;
    }
    std::vector<std::pair<ImageHandle, ImageHandle>> links(pairs.size());
    std::transform(pairs.begin(), pairs.end(), links.begin(), [](const auto& p) {return p->images;});
    mergeModelImpl(src, sim3trans, src2dst, links); // merge and run localBA on all src images.

    pairs.clear();
    return true;
}

void IncreModel::mergeModelImpl(IncreModel& src, const Sim3Transform& transform, 
        const std::unordered_multimap<PointHandle, PointHandle, DirectMappingHash<PointHandle>>& src2dst,
        const std::vector<std::pair<ImageHandle, ImageHandle>>& pairs)
{
    ImgInfoCache dstCache;
    const Eigen::Affine3f sim3 = toAffine3f(transform);
    const Rotation rotation = transform.R;
    VectorMap<PointHandle, PointHandle> ptMap; // handle in src to new handle when merged into dst
    ASSERT(src.mPendingPairs.empty() && src.mUnusedPairs.empty());
    for (auto iter = src.mPoints.begin(); iter != src.mPoints.end(); iter++)
    {
        const PointHandle hOrig = iter.key();
        const PointHandle hNew = mPointHandleGenerator.make();
        ASSERT(ptMap.try_emplace(hOrig, hNew).second);
        Point ptNew = *iter;
        ptNew.location = fromEigen((sim3 * toEigen((*iter).location).homogeneous()).eval());
        ASSERT(mPoints.try_emplace(hNew, ptNew).second);
    }
	const bool rollVGlb = isRollVGlb(mBuilder->config().shutterType);
    for (auto iter = src.mPoses.begin(); iter != src.mPoses.end(); iter++)
    {
        const Pose p = *iter;
        const Pose poseNew{{p.R * rotation.conjugate()}, fromEigen((sim3 * toEigen(p.C).homogeneous()).eval()), rollVGlb ? rotation * p.v : p.v};
        // assertion may fail if we support sync'ed multi-camera in the future.
        // In that case, we check poseNew should be close to the existing one.
        ASSERT(mPoses.try_emplace(iter.key(), poseNew).second);
    }
    for (auto iter = src.mCameras.begin(); iter != src.mCameras.end(); iter++)
    {
        if (!mCameras.has(iter.key())) {
            ASSERT(mCameras.try_emplace(iter.key(), *iter).second);
        }
        else {
            /*@fixme: should match existing one in dst*/
        }
    }
    for (const auto& hImg : src.mFusedImgSeq)
    {
		const auto& img = *mBuilder->getImage(hImg);
        makeIdxObOffsetForImage(hImg, img.nbKPoints, img.getNbTiePtMeasurements());
        mFusedImgSeq.emplace_back(hImg);
        VectorMap<IdxKPt, PointHandle> obPts = std::move(src.mObPoints.at(hImg));
        for(PointHandle& h : obPts) {
            h = ptMap.at(h);
        }
        ASSERT(mObPoints.try_emplace(hImg, std::move(obPts)).second);
        mPendingLocalBA.emplace_back(hImg);
    }
    for (auto iter = src.mPoints.begin(); iter != src.mPoints.end(); iter++)
    {
        const PointHandle hOrig = iter.key();
        const PointHandle hNew = ptMap.at(hOrig);
        for (auto& idxOb : mPoints.at(hNew).observations) {
            const Observation ob = src.decodeIdxOb(idxOb);
            idxOb = encodeIdxOb(ob);
        }
    }
	for (const auto [hPointSrc, hTiePt] : src.mTiePts) {
		const PointHandle hNew = ptMap.at(hPointSrc);
		auto iter = mTiePts.find(hNew);
		if (iter != mTiePts.end()) {
			ASSERT(iter->second == hTiePt);
		}
		else {
			ASSERT(mTiePts.try_emplace(hNew, hTiePt).second);
		}
		if (isAligned()) {
			const auto pCtrlLoc = mBuilder->getCtrlPoint(hTiePt);
			if (pCtrlLoc != nullptr && pCtrlLoc->cov.isFixed()) {
				mPoints.at(hNew).location = pCtrlLoc->loc;
			}
		}
	}
    for (auto iter = src.mCovisibility.begin(); iter != src.mCovisibility.end(); iter++) {
        mCovisibility.try_emplace(iter.key(), *iter);
    }
    for (const auto& p : pairs)
    {
        mCovisibility.at(p.first).emplace_back(p.second);
        mCovisibility.at(p.second).emplace_back(p.first);
    }
    std::unordered_map<PointHandle, PointHandle, DirectMappingHash<PointHandle>> dstMergeMap; // consider using union-find instead
	auto getDstHandle = [&](const PointHandle h) {
		assert(mPoints.has(h) == (dstMergeMap.count(h) == 0));
		if (mPoints.has(h)) {
			assert(dstMergeMap.find(h) == dstMergeMap.end());
			return h;
		}
		PointHandle dst = dstMergeMap.at(h);
		while (!mPoints.has(dst)) [[unlikely]] {
			const PointHandle old = dst; unused(old);
			dst = dstMergeMap.at(dst);
			assert(dst < old); // this guarantees that the loop is finite
		}
		return dst;
	};
    for (auto iter = src2dst.begin(); iter != src2dst.end();)
    {
        const PointHandle hSrc = iter->first;
        const auto range = src2dst.equal_range(hSrc);
        assert(iter == range.first && range.first != range.second);
        const auto iterDst = std::min_element(range.first, range.second, [](const auto& a, const auto& b){
            return a.second < b.second;});
        assert(iterDst != range.second);
        const PointHandle hDst = getDstHandle(iterDst->second);
        mergePoints(hDst, ptMap.at(hSrc), dstCache);
        while (iter != range.second)
        {
            const PointHandle hMergeSrc = getDstHandle(iter->second);
            if (hMergeSrc != hDst) {
                mergePoints(hDst, hMergeSrc, dstCache);
                dstMergeMap.try_emplace(hMergeSrc, hDst);
            }
            iter++;
        }
    }
	for (const auto hPose : src.mGnssPose) {
		mGnssPose.insert(hPose);
	}
#if ENABLE_SANITY_CHECK
    sanityCheck();
#endif
    retriangulate(); // @fixme: only triangulate those involving images in mPendingLocalBA.
    localBA(dstCache);
}
#pragma GCC diagnostic pop

bool IncreModel::tryAlign() {
	ASSERT(!isAligned());
	const auto nbCtrlPts = getNbCtrlPts();
	if (nbCtrlPts + mGnssPose.size() < 3) {
		return false;
	}
	const auto ctrlPts = getCtrlPts();
	const auto poseGnss = getPoseGnss();
	std::vector<const LocCtrl*> locCtrls;
	locCtrls.reserve(ctrlPts.size() + poseGnss.size());
	std::transform(ctrlPts.begin(), ctrlPts.end(), std::back_inserter(locCtrls), [](const auto& x){return x.second;});
	std::transform(poseGnss.begin(), poseGnss.end(), std::back_inserter(locCtrls), [](const auto& x){return x.second;});
	assert(locCtrls.size() == ctrlPts.size() + poseGnss.size());
	std::array<uint32_t, 3> indices{{0u, 1u, 2u}};
	float bestSingularValueProd = 0.f;
	Eigen::Vector2f bestSingularValues{};
	bestSingularValues << 1.f, 0.f;
	Eigen::Matrix3f pts; // each row is a point
	auto checkTriplet = [&](uint32_t i, uint32_t j, uint32_t k) mutable {
		const Eigen::Matrix3f cov = computeCovariance(pts);
		const Eigen::Vector3f singularValues = cov.jacobiSvd().singularValues();
		const float prod = singularValues.topRows<2>().prod();
		if (prod > bestSingularValueProd) {
			indices = {i, j, k};
			bestSingularValueProd = prod;
			bestSingularValues = singularValues.topRows<2>();
		}
	};
	if (locCtrls.size() < 12) {
		for (uint32_t i = 0; i < locCtrls.size(); i++) { // this causes high complexity for linear flight route
			pts.row(0) = toEigen(locCtrls[i]->loc).transpose();
			for (uint32_t j = i + 1; j < locCtrls.size(); j++) {
				pts.row(1) = toEigen(locCtrls[j]->loc).transpose();
				for (uint32_t k = j + 1; k < locCtrls.size(); k++) {
					pts.row(2) = toEigen(locCtrls[k]->loc).transpose();
					checkTriplet(i, j, k);
				}
			}
		}
	}
	else {
		std::mt19937_64 rng{locCtrls.size()};
		std::uniform_int_distribution<uint32_t> dist{0u, cast32u(locCtrls.size() - 1)};
		for (uint32_t n = 0; n < 256u; n++) {
			uint32_t i, j, k;
			i = dist(rng);
			do { j = dist(rng); } while (j == i);
			do { k = dist(rng); } while (k == i || k == j);
			pts.row(0) = toEigen(locCtrls[i]->loc).transpose();
			pts.row(1) = toEigen(locCtrls[j]->loc).transpose();
			pts.row(2) = toEigen(locCtrls[k]->loc).transpose();
			checkTriplet(i, j, k);
		}
	}
	if (bestSingularValues[1] < bestSingularValues[0] * square(1.f / 16)) {
		return false;
	}
	Eigen::Matrix3f srcPts;
	for (uint32_t i = 0; i < 3; i++) {
		const uint32_t idx = indices[i];
		pts.row(i) = toEigen(locCtrls[idx]->loc).transpose();
		if (idx < ctrlPts.size()) {
			const auto& [hPt, hLocCtrl] = ctrlPts[idx];
			srcPts.row(i) = toEigen(mPoints.at(hPt).location).transpose();
			assert(mBuilder->getCtrlPoint(mTiePts.at(hPt)) == hLocCtrl);
		}
		else {
			const auto& [hPose, hLocCtrl] = poseGnss.at(idx - ctrlPts.size());
			srcPts.row(i) = toEigen(mPoses.at(hPose).C).transpose();
			assert(mBuilder->getPoseGNSS(hPose) == hLocCtrl);
		}
	}
	const Sim3Transform trans = findSim3(srcPts, pts);
	transform(trans);
	mIsAligned = true;
	for (const auto& [hPoint, pLocCtrl] : getCtrlPts()) {
		if (pLocCtrl->cov.isFixed()) {
			auto& p = mPoints.at(hPoint);
			p.location = pLocCtrl->loc;
		}
	}
	for (PoseHandle hPose : mGnssPose) {
		const auto pGnss = mBuilder->getPoseGNSS(hPose);
		if (pGnss->cov.isFixed()) {
			mPoses.at(hPose).C = pGnss->loc;
		}
	}
	sanifyRollingShutterVelocity();
		
	return isAligned();
}

void IncreModel::sanifyRollingShutterVelocity() {
	ASSERT(isAligned());
	const auto shutter = mBuilder->config().shutterType;
	if (shutter == ShutterType::kRollingFixedVelocity || (shutter == ShutterType::kRolling1D) || (shutter == ShutterType::kRolling1DLoc)) {
		for (auto iter = mPoses.begin(); iter != mPoses.end(); iter++) {
			const PoseHandle hPose = iter.key();
			auto& p = *iter;
			const auto ref = mBuilder->getPose(hPose)->v;
			if (shutter == ShutterType::kRollingFixedVelocity) {
				p.v = ref;
			}
			else {
				p.v = ref * (ref.dot(p.v) / ref.squaredNorm());
			}
		}
	}
}

std::vector<std::pair<PointHandle, const LocCtrl*>> IncreModel::getCtrlPts() const {
	std::vector<std::pair<PointHandle, const LocCtrl*>> ret;
	for (const auto& [hPt, hTiePt] : mTiePts) {
		const LocCtrl* pLocCtrl = mBuilder->getCtrlPoint(hTiePt);
		if (pLocCtrl != nullptr) {
			ret.emplace_back(hPt, pLocCtrl);
		}
	}
	return ret;
}

uint32_t IncreModel::getNbCtrlPts() const {
	return cast32u(std::count_if(mTiePts.begin(), mTiePts.end(),
		[this](const std::pair<PointHandle, TiePtHandle>& x){
			return mBuilder->getCtrlPoint(x.second) != nullptr;
		}));
}

std::vector<std::pair<PoseHandle, const LocCtrl*>> IncreModel::getPoseGnss() const {
	std::vector<std::pair<PoseHandle, const LocCtrl*>> ret;
	for (const auto hPose : mGnssPose) {
		const LocCtrl* pLocCtrl = mBuilder->getPoseGNSS(hPose);
		ret.emplace_back(hPose, pLocCtrl);
	}
	return ret;
}

void IncreModel::retriangulate() {
    ImgInfoCache cache{};
    std::copy_if(std::make_move_iterator(mUnderConstructedPairs.begin()), std::make_move_iterator(mUnderConstructedPairs.end()), std::back_inserter(mWellConstructedPairs), [this](auto const& p){return isWellConstructed(*p);});
    mUnderConstructedPairs.erase(std::remove_if(mUnderConstructedPairs.begin(), mUnderConstructedPairs.end(), [this](auto const& p){return p == nullptr;}), mUnderConstructedPairs.end());
    for (auto const& p : mUnderConstructedPairs) {
        retriangulatePair(cache, *p, getMatchingSolution(*p));
    }
}

void IncreModel::retriangulatePair(ImgInfoCache& cache, ImagePair const& pair, int32_t idxSolution) {
    int32_t const idxSol = idxSolution == -1 ? getMatchingSolution(pair) : idxSolution;
    float const ratio = getConstructedMatchRatio(pair, idxSol);
    if (ratio > mBuilder->config().model.goodPairConstructionThreshold) {
        return;
    }
    auto const& [hImg0, hImg1] = pair.images;
    if ((uint32_t)hImg0 == 627 || (uint32_t)hImg1 == 627){
        printf("retriangulating %u-%u\n", (uint32_t)hImg0, (uint32_t)hImg1);
    }
    auto& obPts0 = mObPoints.at(hImg0);
    auto& obPts1 = mObPoints.at(hImg1);
    auto const& solution = pair.solutions.at(idxSol);
    auto const& matches = pair.kptsMatches;
    for (auto const idxMatch : solution.inliers) {
        auto const& [idxKPt0, idxKPt1] = matches.at(idxMatch);
        auto const iter0 = obPts0.find(idxKPt0);
        auto const iter1 = obPts1.find(idxKPt1);
        bool const ob0HasPt = (iter0 != obPts0.end());
        bool const ob1HasPt = (iter1 != obPts1.end());
        if (ob0HasPt && ob1HasPt) {
            // Follow what colmap does so far, i.e. never merge points with retriangulation.
            // @fixme: try to allow merging and allow splitting later
            continue;
        }
        PointHandle hPoint = kInvalid<PointHandle>;
        if (!ob0HasPt && !ob1HasPt) {
            // create new point
            hPoint = mPointHandleGenerator.make();
            ASSERT(mPoints.try_emplace(hPoint, nullPoint).second);
        }
        else {
            assert((ob0HasPt || ob1HasPt) && (ob0HasPt != ob1HasPt));
            hPoint = ob0HasPt ? *iter0 : *iter1;
        }
        auto& obs = mPoints.at(hPoint).observations;
        if (!ob0HasPt) {
            obs.emplace_back(encodeIdxOb(Observation{hImg0, idxKPt0}));
            obPts0.try_emplace(idxKPt0, hPoint);
        }
        if (!ob1HasPt) {
            obs.emplace_back(encodeIdxOb(Observation{hImg1, idxKPt1}));
            obPts1.try_emplace(idxKPt1, hPoint);
        }
        bool const isUpdated = updatePointWithLastOb(hPoint, cache);
        unused(isUpdated);
        assert(std::isfinite(mPoints.at(hPoint).avgSqrErr));
    }
}

int32_t IncreModel::getMatchingSolution(ImagePair const& p) const {
    ASSERT(!p.solutions.empty());
    auto const& pose0 = mPoses.at(mBuilder->getImage(p.images.first)->hPose);
    auto const& pose1 = mPoses.at(mBuilder->getImage(p.images.second)->hPose);
    auto invR = pose0.R * pose1.R.conjugate();
    float minAbsDiff = INFINITY;
    int32_t idxMinDiff = 0;
    for (int32_t i = 0; i < (int32_t)p.solutions.size(); i++) {
        auto const diff = p.solutions.at(i).transform.R * invR;
        float const absDiffAngle = std::abs(2*atan2f(Vector3f::Map(&diff.x).norm(), diff.w));
        if (absDiffAngle < minAbsDiff) {
            minAbsDiff = absDiffAngle;
            idxMinDiff = i;
        }
    }
    return idxMinDiff;
}
float IncreModel::getConstructedMatchRatio(ImagePair const& p, int32_t idxSolution) const {
    auto const& [hImg0, hImg1] = p.images;
    auto const& obPts0 = mObPoints.at(hImg0);
    auto const& obPts1 = mObPoints.at(hImg1);
    int32_t const idxSol = idxSolution == -1 ? getMatchingSolution(p) : idxSolution;
    auto const& solution = p.solutions.at(idxSol);
    if (solution.inliers.size() < 8) {
        return 1.f;
    }
    auto const& matches = p.kptsMatches;
    uint32_t nbConstructed = 0;
    for (auto const idxMatch : solution.inliers) {
        auto const& [idxKPt0, idxKPt1] = matches.at(idxMatch);
        auto const iter0 = obPts0.find(idxKPt0);
        if (iter0 == obPts0.end()) {
            continue;
        }
        auto const iter1 = obPts1.find(idxKPt1);
        if (iter1 == obPts1.end()) {
            continue;
        }
        if (*iter0 == *iter1) {
            nbConstructed++;
        }
    }
    return nbConstructed / float(solution.inliers.size());
}

bool IncreModel::isWellConstructed(ImagePair const& p) const {
    return getConstructedMatchRatio(p) >= mBuilder->config().model.goodPairConstructionThreshold;
}

} // namespace rsfm
