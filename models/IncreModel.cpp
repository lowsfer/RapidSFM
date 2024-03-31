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
#include <eigen3/Eigen/Geometry>
#include "../TypesExt.hpp"
#include "../Config.h"
#include "../Builder.h"
#include "../legacy/geometry.hpp"
#include "../Image.h"
#include <eigen3/Eigen/Dense>
#include <RapidSift.h>
#include <DefaultCacheableObject.h>
#include "../ransac.hpp"
#include "FiberUtils.h"
#include "PnPOptimizer.h"
#include <RapidBA.h>
#include "IncreModel.hpp"
#include <algorithm>
#include "../distortion.h"
#include "rbaUtils.h"
#include <Profiler.h>
#include <algorithm>
#include "../rsm.hpp"
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cmath>

#define USE_DELEGATED_STREAM_SYNC 0
#define DEBUG_TRY_ADD_IMAGE_IMPL 0
#define DEBUG_TRY_ADD_IMAGE 0

namespace rsfm{
// Defined in ImagePair.cpp
template <bool isTiePt>
std::array<Eigen::ArrayXf, 4> makePtPair(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, bool undistort);
template <bool isTiePt>
std::array<Eigen::ArrayXf, 4> makePtPair(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, const RealCamera& camFirst, const RealCamera& camSecond);

const rsfm::IModel* toInterface(const IncreModel* m){return m;}

IncreModel::IncreModel(Builder& builder)
    : mBuilder{&builder}
    // Do not create mBundleModel here as user may need to update config before start.
{}
IncreModel::~IncreModel() = default;

void IncreModel::collectUnusedPairs(std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>& unusedPairs)
{
    for (auto& item : mUnusedPairs) {
        ASSERT(!hasFused(item.first));
        for (auto& p: item.second){
            unusedPairs[item.first].emplace_back(std::move(p));
        }
        item.second.clear();
    }
    mUnusedPairs.clear();
}

float IncreModel::checkAffinity(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs)
{
    unused(hImage);
    assert(std::all_of(allPairs.begin(), allPairs.end(), [hImage](const std::unique_ptr<ImagePair>& p){return p->images.second == hImage;}));
    if (!isInitialized()) {
        return 0.f;
    }
    uint32_t totalNbCommonPts = 0u;
    auto detectCommonPts = [&totalNbCommonPts, this](const std::unique_ptr<ImagePair>& pair){
        const ImageHandle hFirst = pair->images.first;
        if (hasFused(hFirst)) {
            const auto& ob2Pt = mObPoints.at(hFirst);
            const uint32_t nbCommonPts = std::count_if(pair->kptsMatches.begin(), pair->kptsMatches.end(),
                [&ob2Pt](const std::pair<IdxKPt, IdxKPt>& m){return ob2Pt.has(m.first);});
            totalNbCommonPts += nbCommonPts;
        }
    };
    for (const auto& pair : allPairs) {
        detectCommonPts(pair);
    }
    assert(mPendingPairs.count(hImage) == 0 && mUnusedPairs.count(hImage) == 0 && !hasFused(hImage));
    const auto& cfg = mBuilder->config();
    if (totalNbCommonPts < std::max(cfg.pnpSolver.minNbInliers, cfg.model.minNbCommonPoints)) {
        return -1.f;
    }
    return static_cast<float>(totalNbCommonPts);
}

bool IncreModel::addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>>& allPairs)
{
#if DEBUG_TRY_ADD_IMAGE
    std::stringstream ss;
    for (auto& p : allPairs) {
        auto const hFirst = p->images.first;
        bool const isFirstFused = hasFused(hFirst);
        if (isFirstFused)
        {
            ss << " \033[1;31m" << (uint32_t)hFirst << "\033[0m";
        }
        else {
            ss << " " << (uint32_t)hFirst;
        }
        
    }
    printf("[Info]: tryAddImage() %u with%s\n", (uint32_t)hImage, ss.str().c_str());
#endif
    if (mPendingPairs.count(hImage) != 0){
        assert(isInitialized() && &allPairs == &mPendingPairs.at(hImage));
    }
    assert(mUnusedPairs.count(hImage) == 0 && !hasFused(hImage));
    assert(std::all_of(allPairs.begin(), allPairs.end(), [hImage](const std::unique_ptr<ImagePair>& p){return p->images.second == hImage;}));
    if (!isInitialized()) {
        mPendingSeq.emplace_back(hImage);
        ASSERT(mPendingPairs.try_emplace(hImage, std::move(allPairs)).second);
		const auto& img = *mBuilder->getImage(hImage);
        makeIdxObOffsetForImage(hImage, img.nbKPoints, img.getNbTiePtMeasurements());
        allPairs = std::vector<std::unique_ptr<ImagePair>>{};
        const bool initialized = tryInitializeModelImpl();
        if (initialized) {
            ASSERT(isInitialized());
            while (!mPendingSeq.empty()) {
                assert(mPendingSeq.size() == mPendingPairs.size());
                std::vector<ImageHandle> fused;
                for (const ImageHandle h : mPendingSeq) {
                    const bool isFused = addImage(h, mPendingPairs.at(h)); // @fixme: use tryAddImageImpl?
                    assert(!isFused || mPendingPairs.at(h).empty());
                    if (isFused) {
                        fused.emplace_back(h);
                    }
                }
                if (fused.empty()) {
                    break;
                }
                for (const ImageHandle h : fused) {
                    const auto pairs = takePendingPairs(h);
                    ASSERT(pairs.empty());
                }
            }
        }
        return true;
    }

    const bool isFused = tryAddImageImpl(hImage, allPairs);
    if (!isFused) {
        return false;
    }

    assert(isFused);
    assert(hasFused(hImage));
    for (std::unique_ptr<ImagePair>& pair : allPairs) {
        std::unique_ptr<ImagePair> pairInv = std::make_unique<ImagePair>(pair->inverse());
        pair.reset();
        assert(pairInv->images.first == hImage);
        const ImageHandle hTarget = pairInv->images.second;
        if (mPendingPairs.find(hTarget) == mPendingPairs.end()) {
            assert(hasFused(pairInv->images.first) && !hasFused(pairInv->images.second));
            mUnusedPairs[hTarget].emplace_back(std::move(pairInv));
        }
        else {
            mPendingPairs.at(hTarget).emplace_back(std::move(pairInv));
        }
    }
    allPairs.clear();
    return true;
}


namespace
{
Eigen::Isometry3f solvePnPRansac(const std::array<Eigen::ArrayXf, 3>& pts3d, const std::array<Eigen::ArrayXf, 2>& pts2d,
	uint32_t nbKPoints, uint32_t nbTiePts,
    const PinHoleCamera& cam, float minSampleSpan, float threshold, float confidence, size_t maxNbTests)
{
	const uint32_t collectionSize = cast32u(pts3d.at(0).rows());
	ASSERT(collectionSize == nbKPoints + nbTiePts);
    auto ransac_test = [&minSampleSpan, &pts3d, &pts2d, &cam, &threshold](const std::array<uint32_t, 4>& indices) -> size_t{
        const Eigen::Matrix<float, 4, 3> samples3d = sampleWithIndices(pts3d, indices);
        const Eigen::Matrix<float, 4, 2> samples2d = sampleWithIndices(pts2d, indices);
#if 0 // should be covered by coveriance check.
        // 2d points should not be too close
        for(int i = 0; i < int(indices.size()); i++){
            for(int j = 0; j < i; j++){
                Eigen::Matrix<float, 1, 4> diff = samples2d.row(i) - samples2d.row(j);
                if(diff.squaredNorm() < minSampleSpan) {
                    return 0u;
                }
            }
        }
#endif
        // the first three 2d points should not be co-linear
        const Eigen::Matrix<float, 3, 2> p3p2d = samples2d.template topRows<3>();
        if (std::sqrt(computeCovariance(p3p2d).jacobiSvd().singularValues().minCoeff()) < minSampleSpan) {
            return 0u;
        }

        const Eigen::Isometry3f trans = solvePnP(samples3d, samples2d, cam);
        const size_t nbInliers = static_cast<size_t>(checkPnP(trans, pts3d, pts2d, cam, threshold).count());
        return nbInliers;
    };
	const uint32_t tiePtSampleFreqRatio = clamp(nbKPoints / 16u, 1u, 256u);;
    const std::array<uint32_t, 4> indicesBest = ransac<decltype(ransac_test), 4>(ransac_test, collectionSize, confidence, maxNbTests, std::random_device{}(), nbTiePts, tiePtSampleFreqRatio);
    return solvePnP(sampleWithIndices(pts3d, indicesBest), sampleWithIndices(pts2d, indicesBest), cam);
}
}// unnamed namespace

//@fixme: add test
bool IncreModel::tryInitializeModelImpl()
{
    ASSERT(!isInitialized());
    if (mPendingSeq.size() < 3){
        return false;
    }
    const ImageHandle hBridge = mPendingSeq.back();
    auto& pairs = mPendingPairs.at(hBridge);
    if (pairs.size() < 2) {
        return false;
    }

    const auto& cfg = mBuilder->config();

    for (uint32_t j = 0; j < pairs.size(); j++) {
        if (mPendingPairs.count(pairs.at(j)->images.first) == 0) {
            continue;
        }
        for (uint32_t i = 0; i < j; i++) {
            if (mPendingPairs.count(pairs.at(i)->images.first) == 0) {
                continue;
            }
            const uint32_t idxP0 = pairs.at(i)->images.first < pairs.at(j)->images.first ? i : j;
            const uint32_t idxP1 = pairs.at(i)->images.first < pairs.at(j)->images.first ? j : i;
            const ImagePair& p0 = *pairs.at(idxP0);
            const ImagePair& p1 = *pairs.at(idxP1);
            assert(p0.images.second == hBridge && p1.images.second == hBridge);
            if (p0.solutions.front().inliers.size() < cfg.model.minPairInliersForInit || p1.solutions.front().inliers.size() < cfg.model.minPairInliersForInit) {
                continue;
            }
            auto isDepthAngleOK = [&cfg](ImagePair const& p){
                // @fixme: move these two constants into config
                return p.maxMedianDepth < cfg.model.maxMedianDepthForInit && p.minMedianAngle > cfg.model.minMedianAngleForInit;
            };
            if (!isDepthAngleOK(p0) || !isDepthAngleOK(p1)) {
                continue;
            }
            const ImageHandle hFirst = p0.images.first;
            const ImageHandle hSecond = p1.images.first;
            assert(hFirst < hSecond);
            assert(std::find_if(mPendingPairs.at(hFirst).begin(), mPendingPairs.at(hFirst).end(), [hSecond](const std::unique_ptr<ImagePair>& p){
                return p->images.first == hSecond;
            }) == mPendingPairs.at(hFirst).end()); // There should be no inversed pair before successful initialization
            const auto iter2 = std::find_if(mPendingPairs.at(hSecond).begin(), mPendingPairs.at(hSecond).end(), [hFirst](const std::unique_ptr<ImagePair>& p){
                return p->images.first == hFirst;
            });
            if (iter2 == mPendingPairs.at(hSecond).end()) {
                continue;
            }
            const ImagePair& p2 = **iter2;
            ASSERT(p2.images.first == hFirst && p2.images.second == hSecond);
            if (p2.solutions.front().inliers.size() < cfg.model.minPairInliersForInit) {
                continue;
            }
            if (!isDepthAngleOK(p2)) {
                continue;
            }
            std::vector<std::pair<std::array<uint32_t, 3>, float>> solutionConsistencyErrors;
            solutionConsistencyErrors.reserve(p0.solutions.size() * p1.solutions.size() * p2.solutions.size());
            for (uint32_t i0 = 0; i0 < p0.solutions.size(); i0++){
                for (uint32_t i1 = 0; i1 < p1.solutions.size(); i1++){
                    for (uint32_t i2 = 0; i2 < p2.solutions.size(); i2++){
                        const Rotation rotErr = p2.solutions[i2].transform.R.conjugate() * p1.solutions[i1].transform.R.conjugate() * p0.solutions[i0].transform.R;
                        const float angleErr = Eigen::AngleAxisf{toEigen(rotErr)}.angle();
                        // printf("angleErr = %f\n", angleErr / M_PI * 180);
                        const float err = std::exp(angleErr * 16.f) / float(p2.solutions[i2].inliers.size() * p1.solutions[i1].inliers.size() * p0.solutions[i0].inliers.size());
                        solutionConsistencyErrors.emplace_back(std::array<uint32_t, 3>{{i0, i1, i2}}, err);
                    }
                }
            }
            if (!solutionConsistencyErrors.empty()) {
                const Image* pFirst = mBuilder->getImage(hFirst);
                const Image* pSecond = mBuilder->getImage(hSecond);
                const auto iterMinErr = std::min_element(solutionConsistencyErrors.begin(), solutionConsistencyErrors.end(), [](const auto& a, const auto& b){return a.second < b.second;});
                const auto& idxS = iterMinErr->first;
                const auto checkSolution = [](const ImagePair& pair, uint32_t idxSolution){
                    using Solution = ImagePair::Solution;
                    return pair.solutions.at(idxSolution).inliers.size() > 0.8f *
                    std::max_element(pair.solutions.begin(), pair.solutions.end(), [](const Solution& a, const Solution& b){
                        return a.inliers.size() < b.inliers.size();})->inliers.size();
                };
                HOPE(checkSolution(p0, idxS[0]));
                HOPE(checkSolution(p1, idxS[1]));
                HOPE(checkSolution(p2, idxS[2]));
				// model is not aligned, yet, so can't use user-provided velocity for rolling shutter here.
				const bool rollVGlb = isRollVGlb(cfg.shutterType);
                ASSERT(mPoses.try_emplace(pFirst->hPose, Pose{Rotation::identity(), {0.f,0.f,0.f}, rollVGlb ? zeroVelocity : mBuilder->getPose(pFirst->hPose)->v}).second);
                const auto& solution = p2.solutions.at(idxS[2]);
                ASSERT(mPoses.try_emplace(pSecond->hPose, (solution.transform * mPoses.at(pFirst->hPose)).withVelocity(rollVGlb ? zeroVelocity : mBuilder->getPose(pSecond->hPose)->v)).second);
                for (const auto& img : {hFirst, hSecond})
                {
                    const auto hCamera = mBuilder->getImage(img)->hCamera;
                    if (mCameras.count(hCamera) == 0u) {
                        ASSERT(mCameras.try_emplace(hCamera, *mBuilder->getRealCamera(hCamera)).second);
                    }
                }
                mObPoints.try_emplace(hFirst);
                mFusedImgSeq.emplace_back(hFirst);
                mObPoints.try_emplace(hSecond);
                mFusedImgSeq.emplace_back(hSecond);
                addNewImgCovisibility(hFirst, nullptr, 0);
                addNewImgCovisibility(hSecond, &hFirst, 1);
                auto& obsFirst = mObPoints.at(hFirst);
                obsFirst.reserve(pFirst->nbKPoints);
                auto& obsSecond = mObPoints.at(hSecond);
                obsSecond.reserve(pSecond->nbKPoints);
                std::vector<std::pair<Index, Index>> matches(solution.inliers.size());
                std::transform(solution.inliers.begin(), solution.inliers.end(), matches.begin(), [&p2](Index idx){return p2.kptsMatches.at(idx);});
				const auto tiePtMatches = p2.tiePtMatches;
				const uint32_t nbAutoPtMatches = cast32u(matches.size());
				const uint32_t nbTiePtMatches = cast32u(tiePtMatches.size());
				const uint32_t nbPtMatches = nbAutoPtMatches + nbTiePtMatches;
                const auto& camFirst = mCameras.at(pFirst->hCamera);
                const auto& camSecond = mCameras.at(pSecond->hCamera);
                const std::array<Eigen::ArrayXf, 4> autoPtPair = makePtPair<false>(mBuilder, *pFirst, *pSecond, matches, camFirst, camSecond);
				const std::array<Eigen::ArrayXf, 4> tiePtPair = makePtPair<true>(mBuilder, *pFirst, *pSecond, tiePtMatches, camFirst, camSecond);
				std::array<Eigen::ArrayXf, 4> ptPair;
				for (int i = 0; i < 4; i++) {
					ptPair[i].resize(nbPtMatches);
					ptPair[i] << autoPtPair[i], tiePtPair[i];
				}
                const Eigen::MatrixX3f pts3d = triangulate(camFirst.pinHole, camSecond.pinHole, toEigen(solution.transform), ptPair);
                Eigen::VectorXf avgReprojSqrErr(nbPtMatches, 1);
                {
                    const Eigen::Matrix<float, 3, 4> projMat2 = toKMat(mCameras.at(pSecond->hCamera).pinHole) * toEigen(solution.transform).matrix().template topRows<3>();
                    Eigen::MatrixXf reprojSqrErr(nbPtMatches, 2);
                    reprojSqrErr <<
                        // left
                        ((pts3d * toKMat(mCameras.at(pFirst->hCamera).pinHole).transpose()).rowwise().hnormalized() - (Eigen::MatrixX2f{nbPtMatches, 2} << ptPair[0], ptPair[1]).finished()).rowwise().squaredNorm(),
                        // right
                        ((pts3d.rowwise().homogeneous() * projMat2.transpose()).rowwise().hnormalized() - (Eigen::MatrixX2f{nbPtMatches, 2} << ptPair[2], ptPair[3]).finished()).rowwise().squaredNorm();
                    const float huberDelta = (std::sqrt(pFirst->width * pFirst->height) + std::sqrt(pSecond->width * pSecond->height)) * 0.5f * cfg.model.triangulationReprojHuberRelative;
                    const float sqrHuberDelta = square(huberDelta);
                    avgReprojSqrErr = (reprojSqrErr.array() < sqrHuberDelta).select(reprojSqrErr, (huberDelta * 2) * reprojSqrErr.array().sqrt() - sqrHuberDelta).rowwise().mean();
                }
                for (uint32_t i = 0; i < nbPtMatches; i++){
                    const PointHandle hPoint = mPointHandleGenerator.make();
					const bool isAutoPtMatch = i < nbAutoPtMatches;
                    const auto& m = isAutoPtMatch ? matches.at(i) : [&]{
						const uint32_t idxTiePt = i - nbAutoPtMatches;
						assert(idxTiePt < nbTiePtMatches); unused(nbTiePtMatches);
						return Pair<IdxKPt>{
							pFirst->nbKPoints + tiePtMatches[idxTiePt].first,
							pSecond->nbKPoints + tiePtMatches[idxTiePt].second
						};
					}();
                    const auto emplaceResult = mPoints.try_emplace(hPoint, Point{
                        Vec3f{pts3d(i, 0), pts3d(i, 1), pts3d(i, 2)},
                        avgReprojSqrErr[i],
                        std::vector<IdxOb>{{encodeIdxOb(Observation{hFirst, m.first}), encodeIdxOb(Observation{hSecond, m.second})}}
                    });
                    ASSERT(emplaceResult.second);
                    obsFirst[m.first] = hPoint;
                    obsSecond[m.second] = hPoint;
					if (!isAutoPtMatch) {
						const uint32_t idxTiePt = i - nbAutoPtMatches;
						const auto& m = tiePtMatches[idxTiePt];
						const TiePtHandle hTiePt = pFirst->tiePtMeasurements.at(m.first).hTiePt;
						assert(pSecond->tiePtMeasurements.at(m.second).hTiePt == hTiePt);
						mTiePts.try_emplace(hPoint, hTiePt);
					}
                }
                if (cfg.model.autoScale && !isAligned()) {
                    float scale = 0.f;
                    for (int i = 0; i < pts3d.rows(); i++){
                        scale += pts3d(i, 2);
                    }
                    scale = 5.f * pts3d.rows() / scale;
                    rescaleModel(scale);
                }

                dbgExpr(sanityCheck());
                mPendingPairs.at(hSecond).erase(iter2);
                for (const ImageHandle hImg : {hFirst, hSecond})
                {
                    if (mPendingPairs.count(hImg) == 0) {
                        continue;
                    }
                    for (auto& p : mPendingPairs.at(hImg)) {
                        assert(p->images.second == hImg);
                        auto pairInv = std::unique_ptr<ImagePair>(new ImagePair{p->inverse()});
                        p.reset();
                        if (mPendingPairs.count(pairInv->images.second) != 0) {
                            mPendingPairs.at(pairInv->images.second).emplace_back(std::move(pairInv));
                        }
                        else {
                            assert(hasFused(pairInv->images.first) && !hasFused(pairInv->images.second));
                            mUnusedPairs[pairInv->images.second].emplace_back(std::move(pairInv));
                        }
                    }
                    unused(takePendingPairs(hImg));
                }
				for (const PoseHandle hPose : {pFirst->hPose, pSecond->hPose}) {
					if (mBuilder->getPoseGNSS(hPose) != nullptr) {
						mGnssPose.insert(hPose);
					}
				}
                mPendingLocalBA.emplace_back(hFirst);
                mPendingLocalBA.emplace_back(hSecond);
#if 1
				if (!isAligned()) {
					const bool success = tryAlign();
					if (success) {
						printf("[Info] Successfully aligned model with the global coordinate system\n");
						ImgInfoCache cache;
						globalBA(cache, false);
						mPendingLocalBA.clear();
					}
				}
#endif
                return true;
            }
        }
    }
    return false;
}


bool IncreModel::tryAddImageImpl(ImageHandle hSecond, std::vector<std::unique_ptr<ImagePair>>& pairs)
{
    const auto trace = cudapp::Profiler::instance().mark(__func__); //__PRETTY_FUNCTION__
    assert(std::all_of(pairs.begin(), pairs.end(), [hSecond](const std::unique_ptr<ImagePair>&p){return p->images.second == hSecond;}));
    assert(!mPoses.has(mBuilder->getImage(hSecond)->hPose));
#if DEBUG_TRY_ADD_IMAGE_IMPL
    std::stringstream ss;
    for (auto& p : pairs) {
        auto const hFirst = p->images.first;
        bool const isFirstFused = hasFused(hFirst);
        if (isFirstFused)
        {
            ss << " \033[1;31m" << (uint32_t)hFirst << "\033[0m";
        }
        else {
            ss << " " << (uint32_t)hFirst;
        }
        
    }
    printf("[Info]: tryAddImageImpl() %u with%s\n", (uint32_t)hSecond, ss.str().c_str());
#endif
    if (std::all_of(pairs.begin(), pairs.end(), [this](const std::unique_ptr<ImagePair>& p){return !hasFused(p->images.first);})){
#if DEBUG_TRY_ADD_IMAGE_IMPL
		printf("[Info]: tryAddImageImpl() %u failed due to lack of fused paring images\n", (uint32_t)hSecond);
#endif
        return false;
    }

    const Image& img = *mBuilder->getImage(hSecond);
    VectorMap<IdxKPt, std::vector<Observation>> observations;
    observations.reserve(img.nbKPoints + img.getNbTiePtMeasurements());
    for (const auto& p : pairs) {
        const ImageHandle hFirst = p->images.first;
        assert(hSecond == p->images.second);
        if (!mPoses.has(mBuilder->getImage(hFirst)->hPose)) {
            continue;
        }
#if 0 // was 1. part of the reason for bug PG-2.
        for (const auto& m : p->kptsMatches) {
            observations[m.second].emplace_back(Observation{hFirst, m.first});
        }
#else
        for (const Index idxInlier : p->solutions.at(0).inliers) {
            const auto& m = p->kptsMatches[idxInlier];
            observations[m.second].emplace_back(Observation{hFirst, m.first});
        }
#endif
		const Image& imgFirst = *mBuilder->getImage(hFirst);
		for (const auto& m : p->tiePtMatches) {
			assert(m.first < imgFirst.getNbTiePtMeasurements() && m.second < img.getNbTiePtMeasurements());
			observations[m.second + img.nbKPoints].emplace_back(Observation{hFirst, m.first + imgFirst.nbKPoints});
		}
    }
    // use PnP to solve the new image pose
    VectorMap<IdxKPt, PointHandle> pnpPts;
    pnpPts.reserve(img.nbKPoints + img.getNbTiePtMeasurements());
    for (auto iter = observations.begin(); iter != observations.end(); iter++) {
        const IdxKPt idxKPt = iter.key();
        PointHandle hPoint = kInvalid<PointHandle>;
        for (const Observation& ob : *iter){
            const auto iterImgMap = mObPoints.find(ob.hImage);
            if (iterImgMap != mObPoints.end()) {
                if (iterImgMap->second.has(ob.idxKPt)) {
                    hPoint = iterImgMap->second.at(ob.idxKPt);
                    break; // @fixme: issue #1 should choose a good 3D point for PnP
                }
            }
        }
        if (hPoint != kInvalid<PointHandle>) {
            pnpPts.try_emplace(idxKPt, hPoint);
        }
    }
    if (pnpPts.size() < std::max(4u, mBuilder->config().pnpSolver.minNbInliers)) {
#if DEBUG_TRY_ADD_IMAGE_IMPL
		printf("[Info]: tryAddImageImpl() %u failed due to insufficient candidate pnp inliers before optimization\n", (uint32_t)hSecond);
#endif
        return false;
    }
    using ArrayXf = Eigen::ArrayXf;
    std::array<ArrayXf, 2> pts2d;
    std::array<ArrayXf, 3> pts3d;
    for (auto& x : pts2d) {
        x.resize(pnpPts.size(), 1);
    }
    for (auto& x : pts3d) {
        x.resize(pnpPts.size(), 1);
    }
    const auto& cam = mCameras.has(img.hCamera) ? mCameras.at(img.hCamera) : *mBuilder->getRealCamera(img.hCamera);
    ImgInfoCache cache;
	uint32_t nbInvolvedKPts = 0u;
	uint32_t nbInvolvedTiePts = 0u;
    {
		const auto& entry = cachedFind(hSecond, cache);
		const auto pt2dGetter = entry.makePt2dGetter();
        uint32_t idxPnpPt = 0;
        for (auto iter = pnpPts.begin(); iter != pnpPts.end(); iter++) {
            const IdxKPt idxKPt = iter.key();
			if (idxKPt < img.nbKPoints) {
				nbInvolvedKPts++;
			}
			else {
				nbInvolvedTiePts++;
			}
            const auto& loc2d = pt2dGetter(idxKPt);
            pts2d[0][idxPnpPt] = loc2d.x;
            pts2d[1][idxPnpPt] = loc2d.y;
            const Vec3f& loc3d = mPoints.at(*iter).location;
            pts3d[0][idxPnpPt] = loc3d.x;
            pts3d[1][idxPnpPt] = loc3d.y;
            pts3d[2][idxPnpPt] = loc3d.z;
            idxPnpPt++;
        }
        ASSERT(idxPnpPt == pnpPts.size());
        undistortInPlace(pts2d.at(0).data(), pts2d.at(1).data(), pnpPts.size(), cam);
    }
	const auto& cfg = mBuilder->config();
    const auto& pnpCfg = cfg.pnpSolver;
    const float imgDim = std::min(img.width, img.height);// std::sqrt(float(img.width * img.height));
    const Eigen::Isometry3f initTrans = solvePnPRansac(pts3d, pts2d, nbInvolvedKPts, nbInvolvedTiePts, cam.pinHole, imgDim * pnpCfg.ransacSampleMinSpan, imgDim * pnpCfg.ransacRelativeThreshold, pnpCfg.requiredRansacConfidence, pnpCfg.maxNbRansacTests);
    const Pose initPose = fromEigen(initTrans).toPose();
    const bool verbose = false;
	const float huberDelta = imgDim * mBuilder->config().model.triangulationReprojHuberRelative;
#if 0
    // RapidBA-based PnP optimizer is deprecated because it's too heavy for such small problem
    const Pose pose = pnpCfg.optimizeWithBA ? mPnPOptimizer->optimize(pts3d, pts2d, cam.pinHole, initPose, huberDelta, verbose) : initPose;
    // const ArrayXf omega = ArrayXf::Constant(pnpPts.size(), 1.f);
    // const ArrayXf huber = ArrayXf::Constant(pnpPts.size(), huberDelta);
    // const Pose fastPose = optimizePnP(cam.pinHole, initPose,
    //     {pts3d.at(0).data(), pts3d.at(1).data(), pts3d.at(2).data()},
    //     {pts2d.at(0).data(), pts2d.at(1).data()},
    //     omega.data(), huber.data(),
    //     pnpPts.size(), 30, 0.001f, verbose);
    // std::cout << "init:\t" << initPose << "\n"
    //     << "rba:\t" << pose << "\n"
    //     << "simd:\t" << fastPose << "\n";
#else
    ArrayXf omega = ArrayXf::Constant(pnpPts.size(), 1.f);
    ArrayXf huber = ArrayXf::Constant(pnpPts.size(), huberDelta);
	std::fill_n(omega.data() + nbInvolvedKPts, nbInvolvedTiePts, mBuilder->config().pnpSolver.tiePtWeight);
	std::fill_n(huber.data() + nbInvolvedKPts, nbInvolvedTiePts, INFINITY);
    Pose pose = optimizePnP(cam.pinHole, initPose,
        {pts3d.at(0).data(), pts3d.at(1).data(), pts3d.at(2).data()},
        {pts2d.at(0).data(), pts2d.at(1).data()},
        omega.data(), huber.data(),
        pnpPts.size(), 30, 0.001f, verbose);
	
    const PoseHandle hPose = img.hPose;
	if (isAligned() || !isRollVGlb(cfg.shutterType)) {
		pose.v = mBuilder->getPose(hPose)->v;
	}
#endif
    const Eigen::Isometry3f trans = toEigen(Transform::fromPose(pose));
    Eigen::Array<bool, Eigen::Dynamic, 1> pnpInlierMask = checkPnP(trans, pts3d, pts2d, cam.pinHole, imgDim * pnpCfg.ransacRelativeThreshold);
	std::fill_n(pnpInlierMask.data() + nbInvolvedKPts, nbInvolvedTiePts, true);
    const size_t nbInliers = pnpInlierMask.count();
#if DEBUG_TRY_ADD_IMAGE_IMPL
    printf("[Info]: pnp: %lu/%lu for image #%u\n", nbInliers, pnpPts.size(), static_cast<uint32_t>(hSecond));
#endif
    if (nbInliers < pnpCfg.minNbInliers || nbInliers < pnpPts.size() * pnpCfg.minInlierRatio) {
#if DEBUG_TRY_ADD_IMAGE_IMPL
		printf("[Info]: tryAddImageImpl() #%u failed due to insufficient inliers after optimization. nbInliers = %u / %u\n", (uint32_t)hSecond, (uint32_t)nbInliers, (uint32_t)pnpPts.size());
#endif
        return false;
    }
    const auto inlierIndices = mask2indices(pnpInlierMask.data(), pnpInlierMask.data() + pnpInlierMask.rows());
    const auto inliers2d = sampleWithIndices(pts2d, inlierIndices.data(), inlierIndices.size());
    if (std::sqrt(computeCovariance(inliers2d).jacobiSvd().singularValues().minCoeff()) < imgDim * pnpCfg.minSpan){
#if DEBUG_TRY_ADD_IMAGE_IMPL
		printf("[Info]: tryAddImageImpl() #%u failed due to insufficient inlier span\n", (uint32_t)hSecond);
#endif
        return false;
    }

    makeIdxObOffsetForImage(hSecond, img.nbKPoints, img.getNbTiePtMeasurements());
    ASSERT(!mPoses.has(hPose));
    ASSERT(mPoses.try_emplace(hPose, pose).second);
    ASSERT(mObPoints.try_emplace(hSecond).second);
    mFusedImgSeq.emplace_back(hSecond);
    if (!mCameras.has(img.hCamera)) {
        mCameras.try_emplace(img.hCamera, cam);
    }

	if (mBuilder->getPoseGNSS(hPose) != nullptr) {
		mGnssPose.insert(hPose);
	}

    // update points
    std::vector<Observation> newObs; // to avoid repeated allocation/free
    std::unordered_set<ImageHandle, DirectMappingHash<ImageHandle>> ptObImages; // to avoid repeated allocation/free
    for (auto iter = observations.begin(); iter != observations.end(); iter++) {
        const IdxKPt idxKPt = iter.key();
        const auto& obs = *iter;
        PointHandle hPoint = kInvalid<PointHandle>;
        bool isMerged = false;
        for (const auto& ob : obs) {
            const auto& obPts = mObPoints.at(ob.hImage);
            if (obPts.has(ob.idxKPt)) {
                if (hPoint == kInvalid<PointHandle>){
                    hPoint = obPts.at(ob.idxKPt);
                }
                else {
                    if (hPoint != obPts.at(ob.idxKPt)) {
                        PointHandle hPointToMerge = obPts.at(ob.idxKPt);
                        if (hPoint > hPointToMerge) {
                            std::swap(hPoint, hPointToMerge);
                        }
                        mergePoints(hPoint, hPointToMerge, cache); // @fixme: be careful about merging
                        isMerged = true;
                    }
                }
            }
        }
        if (hPoint == kInvalid<PointHandle>) {
            hPoint = mPointHandleGenerator.make();
            ASSERT(mPoints.try_emplace(hPoint, nullPoint).second);
        }
        auto& ptObs = mPoints.at(hPoint).observations;
        // copy new observations not in ptObs from obs to newObs;
        newObs = obs; // not including ob in hSecond.
#if 0
        sort(ptObs.begin(), ptObs.end()); // @todo: may replace with assert(std::is_sorted())
        // ASSERT(std::is_sorted(ptObs.begin(), ptObs.end()));
        obs.erase(std::remove_if(obs.begin(), obs.end(), [this, &ptObs](const Observation& ob) -> bool{
            return std::binary_search(ptObs.begin(), ptObs.end(), encodeIdxOb(ob));
        }), obs.end());
#else
        // @fixme: for now we just take the first ob, if a point is measured by one image multiple times
        // In the future, we may change to choosing the best measurement.
        ptObImages.clear();
        std::transform(ptObs.begin(), ptObs.end(), std::inserter(ptObImages, ptObImages.end()), [this](IdxOb ob){return decodeIdxOb(ob).hImage;});
        newObs.erase(std::remove_if(newObs.begin(), newObs.end(), [&ptObImages](const Observation& ob) -> bool{
            return ptObImages.count(ob.hImage);
        }), newObs.end());
#endif
        ptObs.reserve(ptObs.size() + newObs.size() + 1);
        std::transform(newObs.begin(), newObs.end(), std::back_inserter(ptObs), [this](const Observation& ob){return encodeIdxOb(ob);});
        for (const auto& ob : newObs) {
            auto& obPts = mObPoints.at(ob.hImage);
            if (!obPts.has(ob.idxKPt)) {
                obPts.try_emplace(ob.idxKPt, hPoint);
            }
            else {
                assert(obPts.at(ob.idxKPt) == hPoint);
            }
        }
        // observation for hSecond must be the last one (for updatePointWithLastOb())
        ptObs.emplace_back(encodeIdxOb({hSecond, idxKPt}));
        ASSERT(mObPoints.at(hSecond).try_emplace(idxKPt, hPoint).second);
        const bool isUpdated = updatePointWithLastOb(hPoint, cache);

		if (idxKPt >= img.nbKPoints) {
			const TiePtHandle hTiePt = img.tiePtMeasurements.at(idxKPt - img.nbKPoints).hTiePt;
			mTiePts[hPoint] = hTiePt;
		}
        
        if (isMerged || isUpdated || !newObs.empty()) {
            sanifyPointObs(hPoint, cache, square(imgDim * pnpCfg.ransacRelativeThreshold * 4));
        }
        else {
            sanifyLastOb(hPoint, cache, square(imgDim * pnpCfg.ransacRelativeThreshold * 4));
        }
        // std::sort(ptObs.begin(), ptObs.end());
        if (mPoints.has(hPoint)) {
            ptObs.shrink_to_fit();
        }
    }

    {
        std::vector<ImageHandle> covisible;
        covisible.reserve(pairs.size());
        for (const auto& p : pairs) {
            assert(p != nullptr);
            if (hasFused(p->images.first)) {
                covisible.emplace_back(p->images.first);
            }
        }
        addNewImgCovisibility(hSecond, covisible.data(), covisible.size());
    }
    dbgExpr(sanityCheck());
    std::copy_if(std::make_move_iterator(pairs.begin()), std::make_move_iterator(pairs.end()), std::back_inserter(mUnderConstructedPairs), [this](const std::unique_ptr<ImagePair>& p){
        return hasFused(p->images.first);
    });
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(), [this](auto const& p){ return p == nullptr; }), pairs.end());

	if (!isAligned()) {
		const bool success = tryAlign();
		if (success) {
			printf("[Info] Successfully aligned model with the global coordinate system\n");
			globalBA(cache, false);
			mPendingLocalBA.clear();
		}
	}

    mPendingLocalBA.emplace_back(hSecond);

    const auto& baCfg = mBuilder->config().bundle;
    if ((baCfg.applyForFirstThreeImages && mFusedImgSeq.size() == 3)
        || (mFusedImgSeq.size() >= std::max<size_t>(4, mLastGlobalBaNbImages * mBuilder->config().model.ratioGlobalBA))) {
        retriangulate();
		globalBA(cache, false);
		mPendingLocalBA.clear();
        mLastGlobalBaNbImages = mFusedImgSeq.size();
	}

    if (mPendingLocalBA.size() >= mBuilder->config().model.intervalLocalBA) {
        if (mFusedImgSeq.size() <= baCfg.useGlobalIfNoLargerThan || mFusedImgSeq.size() == mPendingLocalBA.size()){
            retriangulate();
            globalBA(cache, false);
            mPendingLocalBA.clear();
            mLastGlobalBaNbImages = mFusedImgSeq.size();
        }
        else {
            retriangulate(); // @fixme: only triangulate those involving images in mPendingLocalBA.
            localBA(cache);
        }
    }
#if DEBUG_TRY_ADD_IMAGE_IMPL
    printf("[Info]: tryAddImageImpl() #%u success\n", (uint32_t)hSecond);
#endif
    return true;
}

void IncreModel::sanifyLastOb(PointHandle hPoint, ImgInfoCache& cache, float sqrErrThres)
{
    const auto trace = cudapp::Profiler::instance().mark(__func__); //__PRETTY_FUNCTION__
	const bool isTiePt = mTiePts.find(hPoint) != mTiePts.end();
	if (isTiePt) {
		return;
	}
    Point& p = mPoints.at(hPoint);
    auto& ptObs = p.observations;
    assert(!ptObs.empty());
    const auto& idxOb = ptObs.back();
    const Observation ob = decodeIdxOb(idxOb);
    const auto& cacheEntry = cachedFind(ob.hImage, cache);
	const auto& pt2d = cacheEntry.keyPoints.data()[ob.idxKPt].location;
    
	const auto& pose = mPoses.at(cacheEntry.hPose);
	const Eigen::Vector3f transLoc = toEigen(pose.transform(p.location, mBuilder->config().shutterType, pt2d.y, cacheEntry.image->getRollingCenter()));
    const Eigen::Vector2f uv = transLoc.hnormalized();
    const auto& cam = mCameras.at(cacheEntry.hCamera);
    const auto& cfg = mBuilder->config();
    const float sqrErr = (toEigen(cam.project(cfg.opticModel, fromEigen(uv))) - Eigen::Vector2f::Map(&pt2d.x)).squaredNorm();
    const bool isValid = (inRange(transLoc[2], cfg.model.zMin, cfg.model.zMax) && sqrErr < sqrErrThres);
    if (!isValid) {
        ptObs.erase(std::prev(ptObs.end()));
        assert(mObPoints.at(ob.hImage).at(ob.idxKPt) == hPoint);
        mObPoints.at(ob.hImage).erase(ob.idxKPt);
    }
    if (ptObs.size() == 1) {
        const Observation ob = decodeIdxOb(ptObs[0]);
        ptObs.clear();
        assert(mObPoints.at(ob.hImage).at(ob.idxKPt) == hPoint);        
        mObPoints.at(ob.hImage).erase(ob.idxKPt);
    }
    if (ptObs.empty()) {
        mPoints.erase(hPoint);
    }
}

void IncreModel::sanifyPointObs(PointHandle hPoint, ImgInfoCache& cache, float sqrErrThres) {
    const auto trace = cudapp::Profiler::instance().mark(__func__); //__PRETTY_FUNCTION__
    Point& p = mPoints.at(hPoint);
    auto& ptObs = p.observations;
    std::sort(ptObs.begin(), ptObs.end());
    ptObs.erase(std::unique(ptObs.begin(), ptObs.end()), ptObs.end());
	assert(ptObs.size() >= 2);
	const auto iterTiePt = mTiePts.find(hPoint);
	const bool isTiePt = (iterTiePt != mTiePts.end());
#ifndef NDEBUG
	{
		auto matchIsTiePt = [&](IdxOb idxOb){
			const Observation ob = decodeIdxOb(idxOb);
			const auto& entry = cachedFind(ob.hImage, cache);
			return entry.isTiePt(ob.idxKPt) == isTiePt;
		};
		assert(std::all_of(ptObs.begin(), ptObs.end(), matchIsTiePt));
	}
#endif
	const bool setHardCtrlPt = isAligned() && isTiePt && checkIsHardCtrlPt(iterTiePt->second);
	if (setHardCtrlPt) {
		p.location = mBuilder->getCtrlPoint(iterTiePt->second)->loc;
		return;
	}
#if 1
    const auto& cfg = mBuilder->config();
    Observation obLastValid = {kInvalid<ImageHandle>, kInvalid<IdxKPt>}; // use to elimiate conflicting observations of the same point in the same image
    size_t nbValidObs = 0;
    bool canTriangulate = false || isTiePt;
    Eigen::Vector3f ray0;
	const bool rollVGlb = isRollVGlb(cfg.shutterType);
    for (size_t i = 0; i < ptObs.size(); i++){
        const auto& idxOb = ptObs[i];
        const Observation ob = decodeIdxOb(idxOb);
        const auto& cacheEntry = cachedFind(ob.hImage, cache);
        const auto& cam = mCameras.at(cacheEntry.hCamera);
		const float2 pt2d = cacheEntry.getPt2d(ob.idxKPt);
        const auto& pose = mPoses.at(cacheEntry.hPose);
        if (i == 0) {
            ray0 = (toEigen(pose.getRollingC(rollVGlb, pt2d.y, cacheEntry.image->getRollingCenter())) - toEigen(p.location)).normalized();
        }
        else if (!canTriangulate){
            const Eigen::Vector3f ray1 = (toEigen(pose.getRollingC(rollVGlb, pt2d.y, cacheEntry.image->getRollingCenter())) - toEigen(p.location)).normalized();
            if (ray0.dot(ray1) < cfg.model.cosMinAngle) {
                canTriangulate = true;
            }
        }
        const Eigen::Vector3f transLoc = toEigen(pose.transform(p.location, cfg.shutterType, pt2d.y, cacheEntry.image->getRollingCenter()));
        const Eigen::Vector2f uv = transLoc.hnormalized();
        const float sqrErr = (toEigen(cam.project(cfg.opticModel, fromEigen(uv))) - Eigen::Vector2f::Map(&pt2d.x)).squaredNorm();
        const bool isValid = isTiePt || (ob.hImage != obLastValid.hImage && inRange(transLoc[2], cfg.model.zMin, cfg.model.zMax) && sqrErr < sqrErrThres);
        if (isValid) {
            obLastValid = ob;
            if (i != nbValidObs) {
                ptObs.at(nbValidObs) = idxOb;
            }
            nbValidObs++;
        }
        else {
            assert(mObPoints.at(ob.hImage).at(ob.idxKPt) == hPoint);
            ptObs[i] = kInvalid<DefaultModel::IdxOb>;
            mObPoints.at(ob.hImage).erase(ob.idxKPt);
        }
    }
    const bool isUpdated = (nbValidObs != ptObs.size());
    ptObs.resize(nbValidObs);
    if (ptObs.size() == 1) {
        const Observation ob = decodeIdxOb(ptObs[0]);
        ptObs.clear();
        assert(mObPoints.at(ob.hImage).at(ob.idxKPt) == hPoint);        
        mObPoints.at(ob.hImage).erase(ob.idxKPt);
    }
    if (!canTriangulate) {
        for (const auto& idxOb : ptObs) {
            const Observation ob = decodeIdxOb(idxOb);
            assert(mObPoints.at(ob.hImage).at(ob.idxKPt) == hPoint);
            mObPoints.at(ob.hImage).erase(ob.idxKPt);
        }
        ptObs.clear();
    }
    if (ptObs.empty()) {
        mPoints.erase(hPoint);
    }
    else if (isUpdated){
        assert(mPoints.has(hPoint));
        updatePointWithLastOb(hPoint, cache);
        sanifyPointObs(hPoint, cache, sqrErrThres);
    }
#endif
#if 0
    sanityCheck();
#endif 
}

#pragma GCC diagnostic push
// #pragma GCC optimize ("O0")
void IncreModel::mergePoints(PointHandle dst, PointHandle src, ImgInfoCache& cache)
{
    // dbgExpr(sanityCheck());
    auto& dstPt = mPoints.at(dst);
    auto& srcPt = mPoints.at(src);
    auto& dstPtObs = dstPt.observations;
    auto& srcPtObs = srcPt.observations;
    for (auto idxSrcOb : srcPtObs) {
        dstPtObs.emplace_back(idxSrcOb);
        updatePointWithLastOb(dst, cache);
    }
    for (IdxOb idxSrcOb : srcPtObs) {
        const auto srcOb = decodeIdxOb(idxSrcOb);
        ASSERT(mObPoints.at(srcOb.hImage).at(srcOb.idxKPt) == src);
        mObPoints.at(srcOb.hImage).at(srcOb.idxKPt) = dst;
    }
    mPoints.erase(src);
	const auto iterTiePtSrc = mTiePts.find(src);
	if (iterTiePtSrc != mTiePts.end()) {
		const TiePtHandle hTiePt = iterTiePtSrc->second;
		mTiePts.erase(iterTiePtSrc);
		const auto iterTiePtDst = mTiePts.find(dst);
		if (iterTiePtDst != mTiePts.end()) {
			ASSERT(iterTiePtDst->second == hTiePt);
		}
		else {
			ASSERT(mTiePts.try_emplace(dst, hTiePt).second);
		}
	}
    // dbgExpr(sanityCheck());
}
#pragma GCC diagnostic pop

bool IncreModel::checkIsHardCtrlPt(PointHandle hPoint) const {
	const auto iterTiePt = mTiePts.find(hPoint);
	if (iterTiePt == mTiePts.end()) {
		return false;
	}
	return checkIsHardCtrlPt(iterTiePt->second);
}

bool IncreModel::checkIsHardCtrlPt(TiePtHandle hTiePt) const {
	const auto pLocCtrl = mBuilder->getCtrlPoint(hTiePt);
	return pLocCtrl != nullptr && pLocCtrl->cov.isFixed();
}

bool IncreModel::updatePointWithLastOb(PointHandle hPoint, ImgInfoCache& cache)
{
    Point& p = mPoints.at(hPoint);
    if (std::isfinite(p.avgSqrErr) && p.observations.size() > mBuilder->config().model.nbStableObs) {
        // When we have many observations, the location should have stablized and we don't need to update
        return false;
    }
	const bool isHardCtrlPt = checkIsHardCtrlPt(hPoint);
	if (isHardCtrlPt) {
		const auto opticModel = mBuilder->config().opticModel;
		const auto shutter = mBuilder->config().shutterType;
		p.avgSqrErr = 1.f / p.observations.size() * std::accumulate(p.observations.begin(), p.observations.end(), 0.f,
			[this, &cache, &p, opticModel, shutter](float acc, IdxOb idxOb)->float{
				const Observation ob = decodeIdxOb(idxOb);
				const auto& entry = cachedFind(ob.hImage, cache);
				const auto& pose = mPoses.at(entry.hPose);
				const auto& camera = mCameras.at(entry.hCamera);
				const auto p2d = fromRBA(entry.getPt2d(ob.idxKPt));
				const auto proj2d = camera.project(opticModel, fromEigen((toEigen(pose.transform(p.location, shutter, p2d.y, entry.image->getRollingCenter()))).hnormalized().eval()));
				const float sqrErr = (proj2d - p2d).squaredNorm();
				return acc + sqrErr; // clamp at square(huber*8)
			});
		return false; // @fixme: may return early for all tie points
	}
    const IntriType intriType = mBuilder->config().opticModel;
    using NormXY = Vec2f;
    using Ray = std::pair<const Pose*, NormXY>; // note that NormXY is normalized 3d {x/z, y/z}, not pt2d
    auto makeRay = [&](const Observation& ob)->Ray {
        const CacheEntry& imgInfo = cachedFind(ob.hImage, cache);
        const float2 p2d = imgInfo.getPt2d(ob.idxKPt);
        const InverseRealCamera& camInv = imgInfo.camInv;
        const NormXY uv = camInv.project(intriType, fromRBA(p2d));
        return Ray{&mPoses.at(imgInfo.hPose), uv};
    };
    std::vector<Ray> allRays(p.observations.size());
    struct ProjInfo{
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Matrix<float, 3, 4> projMat;
        Vec2f measurement; // This is the undistorted measurement location computed from rays.
    };
    EigenAlignedVector<ProjInfo> allProjInfo(p.observations.size());
	float imgDim = 0.f;
	const bool rollVGlb = isRollVGlb(mBuilder->config().shutterType);
    for (size_t i = 0; i < p.observations.size(); i++) {
        const Observation& ob = decodeIdxOb(p.observations[i]);
        allRays[i] = makeRay(ob);
        const auto& imgInfo = cachedFind(ob.hImage, cache);
        const auto& normXY = allRays[i].second;
        const auto& pinHole = mCameras.at(imgInfo.hCamera).pinHole;
        const Vec2f measurement {
            pinHole.f.x * normXY.x + pinHole.c.x,
            pinHole.f.y * normXY.y + pinHole.c.y
        };
        allProjInfo[i] = ProjInfo{
            toEigen(Transform::fromRollingPose(mPoses.at(imgInfo.hPose), rollVGlb, imgInfo.getPt2d(ob.idxKPt).y, imgInfo.image->getRollingCenter())).matrix().template topRows<3>().eval(),
            measurement
        };
		imgDim += std::sqrt(imgInfo.image->width * imgInfo.image->height);
    }    
	imgDim /= p.observations.size();

    const Ray& newRay = allRays.back();
    const float huber = imgDim * mBuilder->config().model.triangulationReprojHuberRelative;
    const float avgFactor = 1.f / allProjInfo.size();
    bool isUpdated = false;
    // @fixme: This can be optimized like GEMM using SIMD, in case there is a perf problem
    for (uint32_t i = 0; i + 1 < allRays.size(); i++) {
        const std::array<std::pair<const Pose*, NormXY>, 2> rays = {{newRay, allRays[i]}};
        const Vec3f newLoc = triangulate<2>(rays.data());
        const float avgSqrErr = std::accumulate(allProjInfo.begin(), allProjInfo.end(), 0.f, [&newLoc, huber](float acc, const ProjInfo& proj)->float{
            const float sqrErr = ((proj.projMat * toEigen(newLoc).homogeneous()).hnormalized() - toEigen(proj.measurement)).squaredNorm();
            const float robustSqrErr = (sqrErr < square(huber)) ? sqrErr : (huber * 2 * std::sqrt(sqrErr) - square(huber));
            return acc + std::min(robustSqrErr, square(huber*8)); // clamp at square(huber*8)
        }) * avgFactor;
        if (avgSqrErr < p.avgSqrErr) {
            p.location = newLoc;
            p.avgSqrErr = avgSqrErr;
            isUpdated = true;
        }
    }
    return isUpdated;
}

const IncreModel::CacheEntry& IncreModel::cachedFind(ImageHandle hImage, ImgInfoCache& cache) const
{
    if (cache.count(hImage) == 0) {
        const auto& img = *mBuilder->getImage(hImage);
        const cudaStream_t stream = mBuilder->anyStream();
        auto const iterCam = mCameras.find(img.hCamera);
        rsfm::RealCamera const* const pCam = (iterCam != mCameras.end() ? &*iterCam : mBuilder->getRealCamera(img.hCamera));
        cache.try_emplace(hImage, std::make_unique<CacheEntry>(CacheEntry{
            img.hPose, img.hCamera, &img,
            cudapp::storage::acquireMemory<const KeyPoint>(mBuilder->storageManager(), img.keyPoints, cudapp::storage::StorageLocation::kSysMem, stream, false, true)
            , pCam->inverse()}));
#if USE_DELEGATED_STREAM_SYNC
        // mBuilder->fiberBlockingService()->syncCudaStream(stream);
        cudapp::fiberSyncCudaStream(stream);
#else
        cudaCheck(cudaStreamSynchronize(stream));
#endif
    }
    return *cache.at(hImage);
}

std::vector<std::unique_ptr<ImagePair>> IncreModel::takePendingPairs(ImageHandle hImage)
{
    const auto iterErase = std::find(mPendingSeq.begin(), mPendingSeq.end(), hImage);
    ASSERT(iterErase != mPendingSeq.end());
    mPendingSeq.erase(iterErase);
    std::vector<std::unique_ptr<ImagePair>> pairs = std::move(mPendingPairs.at(hImage));
    mPendingPairs.erase(hImage);
    return pairs;
}

std::vector<std::pair<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>> IncreModel::takeAllPendingPairs()
{
    assert(mPendingSeq.size() == mPendingPairs.size());
    std::vector<std::pair<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>> result(mPendingPairs.size());
    std::transform(mPendingSeq.begin(), mPendingSeq.end(), result.begin(), [this](ImageHandle hImage){
        return std::make_pair(hImage, std::move(mPendingPairs.at(hImage)));
    });
    mPendingSeq.clear();
    mPendingPairs.clear();
    return result;
}

void IncreModel::sanityCheck()
{
    for (auto iter = mPoints.begin(); iter != mPoints.end(); iter++) {
        const Point& p = *iter;
        ASSERT(std::isfinite(p.avgSqrErr));
        for (const IdxOb idxOb : p.observations) {
            const Observation ob = decodeIdxOb(idxOb);
            ASSERT(mObPoints.at(ob.hImage).at(ob.idxKPt) == iter.key());
        }
    }
    for (const auto& item : mObPoints) {
        const ImageHandle hImage = item.first;
        for (auto iter = item.second.begin(); iter != item.second.end(); iter++) {
            const IdxKPt idxKPt = iter.key();
            const PointHandle hPoint = *iter;
            const Point& p = mPoints.at(hPoint);
            ASSERT(std::find(p.observations.begin(), p.observations.end(), encodeIdxOb({hImage, idxKPt})) != p.observations.end());
        }
    }
}

void IncreModel::writePly(const char* filename) const
{
    ASSERT(!mPtColor.empty() || mPoints.empty());
    std::ofstream fout;
    fout.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    fout.open(filename, std::ofstream::trunc);
    fout << "ply\n"
         << "format binary_little_endian 1.0\n"
         << "comment Generated by RapidSFM\n"
         << "element vertex " << mPoints.size() << '\n'
         << "property float x\n"
         << "property float y\n"
         << "property float z\n"
         << "property uchar red\n"
         << "property uchar green\n"
         << "property uchar blue\n"
         << "end_header\n";
    fout.close();
    fout.open(filename, std::ofstream::binary | std::ofstream::app);
    for(auto iter = mPoints.begin(); iter != mPoints.end(); iter++)
    {
        const Point& p = *iter;
        const Color& c = mPtColor.at(iter.key());
        fout.write((const char *) p.location.data(), sizeof(float) * 3);
        fout.write((const char *) &c.r, sizeof(uint8_t) * 3);
    }
    fout.close();
}

void IncreModel::writeNvm(const char* filename) const {
    ASSERT(!mPtColor.empty() || mPoints.empty());

    ImgInfoCache cache;
    std::ofstream fout;
    fout.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    fout.open(filename, std::ofstream::trunc);
    const bool isSharedIntrinsics = (mCameras.size() == 1);
    if (isSharedIntrinsics && !mBuilder->config().bundle.optimizeIntrinsics) {
		const auto iter = mCameras.begin();
		const auto hCam = iter.key();
        const auto& cam = (*iter).pinHole;
		const Vec2<uint32_t> res = mBuilder->getCamResolution(hCam);
        fout << format("NVM_V3 FixedK %f %f %f %f\n") % cam.f.x % (cam.c.x - res.x * 0.5f) % cam.f.y % (cam.c.y - res.y * 0.5f);
    }
    else {
        fout << "NVM_V3\n";
    }
    VectorMap<ImageHandle, uint32_t> imgID2nvmID;
    for (uint32_t i = 0; i < mFusedImgSeq.size(); i++) {
        imgID2nvmID.try_emplace(mFusedImgSeq[i], i);
    }
    fout << mFusedImgSeq.size() << '\n';
    for(const ImageHandle hImg : mFusedImgSeq) {
        const auto& entry = cachedFind(hImg, cache);
        const Pose& pose = mPoses.at(entry.hPose);
		const auto& realCam = mCameras.at(entry.hCamera);
        const PinHoleCamera& cam = realCam.pinHole;
        const Rotation& q = pose.R;
        const Vec3f C = pose.C;
        fout << format("\"%s\" %f %f %f %f %f %f %f %f %f 0\n") % mBuilder->getSavedName(hImg) % ((cam.f.x + cam.f.y) / 2)
                % q.w % q.x % q.y % q.z % C.x % C.y % C.z % realCam.k1();
    }

    fout << mPoints.size() << '\n';
    for(auto iterP = mPoints.begin(); iterP != mPoints.end(); iterP++) {
        const PointHandle hPoint = iterP.key();
        const Point& p = *iterP;
        assert(p.observations.size() >= 2);
#if 0
        fout << format("%f %f %f %d %d %d %d") % p.location.x % p.location.y % p.location.z
            % (int) mPtColor.at(hPoint).r % (int) mPtColor.at(hPoint).g % (int) mPtColor.at(hPoint).b % static_cast<int>(p.observations.size());
#else
        fout << p.location.x << ' ' << p.location.y << ' ' << p.location.z
            << ' ' << (int) mPtColor.at(hPoint).r << ' ' << (int) mPtColor.at(hPoint).g
            << ' ' << (int) mPtColor.at(hPoint).b << ' ' << static_cast<int>(p.observations.size());
#endif
        for (const IdxOb idxOb : p.observations) {
            const Observation ob = decodeIdxOb(idxOb);
            const auto& entry = cachedFind(ob.hImage, cache);
            const float2& pt2d = entry.getPt2d(ob.idxKPt);
			const Vec2f imgCenter = {entry.image->width * 0.5f, entry.image->height * 0.5f};
#if 0
            fout << format(" %d %d %f %f") % imgID2nvmID[ob.hImage] % ob.idxKPt %
                        (pt2d.x - imgCenter.x) % (pt2d.y - imgCenter.y);
#else
            fout << ' ' << imgID2nvmID.at(ob.hImage) << ' ' << ob.idxKPt << ' '
                 << (pt2d.x - imgCenter.x) << ' ' << (pt2d.y - imgCenter.y);
#endif
        }

        fout << '\n';
    }
    fout.close();
}

void IncreModel::writeRsm(const char* filename) const {
    std::ofstream fout;
    fout.exceptions(std::ios::badbit | std::ios::failbit);
    fout.open(filename, std::ios::binary | std::ios::trunc);
    serialize(fout);
}

VectorMap<PointHandle, Color> IncreModel::computePointColor() const
{
    const cudaStream_t stream = mBuilder->anyStream();
    struct ColorAcc
    {
        uint16_t r;
        uint16_t g;
        uint16_t b;
        uint16_t counter;
    };
    static constexpr uint32_t maxColorAccCount = 256u;
    VectorMap<PointHandle, ColorAcc> ptsColorAcc;
    for (const auto& [hImage, obPts] : mObPoints) {
        const Image& img = *mBuilder->getImage(hImage);
        const auto kptsColorHolder = cudapp::storage::acquireMemory<const uchar3>(mBuilder->storageManager(), img.kptsColor, cudapp::storage::StorageLocation::kSysMem, stream, false, true);
        const uchar3* kptsColor = kptsColorHolder.data();
        assert(kptsColorHolder.nbElems() == img.nbKPoints);
#if USE_DELEGATED_STREAM_SYNC
        // mBuilder->fiberBlockingService()->syncCudaStream(stream);
        cudapp::fiberSyncCudaStream(stream);
#else
        cudaCheck(cudaStreamSynchronize(stream));
#endif
        for (auto iter = obPts.begin(); iter != obPts.end(); iter++) {
            const IdxKPt idxKPt = iter.key();
            const uchar3& color = kptsColor[idxKPt];
            const PointHandle hPoint = *iter;
            ColorAcc& colorAcc = ptsColorAcc[hPoint];
            if (colorAcc.counter < maxColorAccCount) {
                colorAcc.r += color.x;
                colorAcc.g += color.y;
                colorAcc.b += color.z;
                colorAcc.counter++;
            }
        }
    }
    VectorMap<PointHandle, Color> ptsColor;
    for (auto iter = ptsColorAcc.begin(); iter != ptsColorAcc.end(); iter++) {
        const PointHandle hPoint = iter.key();
        const ColorAcc& colorAcc = *iter;
        const auto getAvg = [&](uint16_t acc){
            return static_cast<uint8_t>(std::round(static_cast<float>(acc) / colorAcc.counter));
        };
        const Color color {
            getAvg(colorAcc.r), getAvg(colorAcc.g), getAvg(colorAcc.b)
        };
        ASSERT(ptsColor.try_emplace(hPoint, color).second);
    }
    return ptsColor;
}

void IncreModel::addNewImgCovisibility(ImageHandle hImage, const ImageHandle* neighbour, size_t nbNeighbours)
{
    ASSERT(std::all_of(neighbour, neighbour+nbNeighbours, [this](ImageHandle h){return hasFused(h);}));
    ASSERT(mCovisibility.try_emplace(hImage, std::vector<ImageHandle>{neighbour, neighbour+nbNeighbours}).second);
    std::for_each(neighbour, neighbour+nbNeighbours, [this, hImage](ImageHandle h){return mCovisibility.at(h).emplace_back(hImage);});
}

void IncreModel::serialize(std::ostream& ostream) const {
    auto m = getModel();
    rsm::saveRapidSparseModel(m, ostream);
}
rsm::RapidSparseModel IncreModel::getModel() const
{
    VectorMap<PointHandle, Color> newPtColors;
    if (mPtColor.empty()) {
        newPtColors = computePointColor();
    }
    const auto& ptColor = mPtColor.empty() ? newPtColors : mPtColor;
    rsm::RapidSparseModel m{rsm::Model<1>{}};
    auto& v1 = m.get<1>();
    auto fillVector = [](auto& dst, const auto& src, const auto& cvt) {
        using Handle = typename std::decay_t<decltype(src)>::Key;
        VectorMap<Handle, Index> handle2idx;
        for (auto iter = src.begin(); iter != src.end(); iter++){
            ASSERT(handle2idx.try_emplace(iter.key(), static_cast<Index>(dst.size())).second);
            dst.emplace_back(cvt(*iter));
        }
        return handle2idx;
    };
    const auto fwd = [](const auto& x){return x;};
	const auto rollVGlb = isRollVGlb(mBuilder->config().shutterType);
    const VectorMap<PoseHandle, Index> hPose2Idx = rollVGlb ? fillVector(v1.poses, mPoses, fwd) : fillVector(v1.poses, mPoses, [](const Pose& p){
		// const auto v = p.R.conjugate() * -p.v;
		// printf("saving: %f, %f, %f -> %f, %f, %f (* 1E-4)\n", p.v.x*1E4f, p.v.y*1E4f, p.v.z*1E4f, v.x * 1E4f, v.y * 1E4f, v.z*1E4f);
		return Pose{p.R, p.C, p.R.conjugate() * -p.v};
	});
    const VectorMap<CameraHandle, Index> hCamera2Idx = fillVector(v1.cameras, mCameras, fwd);
    const VectorMap<PointHandle, Index> hPoint2Idx = fillVector(v1.points, mPoints,
        [](const auto& x){return x.location;});
    for (auto iter = mPoints.begin(); iter != mPoints.end(); iter++){
        assert(hPoint2Idx.at(iter.key()) == v1.pointColor.size());
        v1.pointColor.emplace_back(ptColor.at(iter.key()));
    }
    ASSERT(v1.points.size() == v1.pointColor.size());
    
    VectorMap<CameraHandle, Vec2<int>> cameraResolutions;
	ImgInfoCache cache;
    for (const auto hImage : mFusedImgSeq) {
        rsm::Capture cap;
        const Image& img = *mBuilder->getImage(hImage);
        if (cameraResolutions.has(img.hCamera)){
            ASSERT(img.width == cameraResolutions.at(img.hCamera).x);
            ASSERT(img.height == cameraResolutions.at(img.hCamera).y);
        }
        else {
            cameraResolutions.try_emplace(img.hCamera, Vec2<int>{img.width, img.height});
        }
        cap.filename = mBuilder->getSavedName(hImage);
        cap.md5sum = img.md5sum;
        cap.idxCamera = hCamera2Idx.at(img.hCamera);
        cap.idxPose = hPose2Idx.at(img.hPose);
        std::vector<rsm::Measurement>& measurements = cap.measurements;
        const VectorMap<IdxKPt, PointHandle>& obPts = mObPoints.at(hImage);
		const auto& entry = cachedFind(hImage, cache);
		auto pt2dGetter = entry.makePt2dGetter();
        for (auto iter = obPts.begin(); iter != obPts.end(); iter++) {
            const IdxKPt idxKPt = iter.key();
            const PointHandle hPoint = *iter;
            const float2 pt2d = pt2dGetter(idxKPt);
            measurements.emplace_back(rsm::Measurement{hPoint2Idx.at(hPoint), {pt2d.x, pt2d.y}});
        }
        v1.captures.emplace_back(std::move(cap));
		cache.clear(); // release cache entry as it will never hit later.
    }
    v1.cameraResolutions.resize(v1.cameras.size());
    for (auto iter = cameraResolutions.begin(); iter != cameraResolutions.end(); iter++) {
        v1.cameraResolutions.at(hCamera2Idx.at(iter.key())) = {cast32u((*iter).x), cast32u((*iter).y)};
    }
    return m;
}

bool IncreModel::isPending(ImageHandle hImage) const {
    assert(mPendingSeq.size() == mPendingPairs.size());
    return mPendingPairs.find(hImage) != mPendingPairs.end();
}

void IncreModel::printSummary(std::ostream& o)
{
    o << makeFmtStr("nbCam = %s, nbCap = %s, nbPt = %s, nbOb = %s\n", mCameras.size(), mObPoints.size(), mPoints.size(),
        std::accumulate(mObPoints.begin(), mObPoints.end(), 0UL, [](size_t acc, const auto& x) {return acc + x.second.size();}));
    if (mCameras.size() == 1) {
        const auto& cam = *mCameras.begin();
        o << makeFmtStr("{fx,fy,cx,cy,k1,k2,p1,p2,k3} = {%f, %f, %f, %f, %f, %f, %f, %f, %f}\n", cam.pinHole.f.x, cam.pinHole.f.y, cam.pinHole.c.x, cam.pinHole.c.y, cam.k1(), cam.k2(), cam.p1(), cam.p2(), cam.k3());
    }
#if 0
    if (!mUnderConstructedPairs.empty()) {
        o << "Under-constructed pairs:\n";
        for (auto const& p : mUnderConstructedPairs) {
            auto const [hFirst, hSecond] = p->images;
            auto const idxSol = getMatchingSolution(*p);
            o << makeFmtStr("%u - %u: %f x %u\n", (uint32_t)hFirst, (uint32_t)hSecond, getConstructedMatchRatio(*p, idxSol), (unsigned)p->solutions.at(idxSol).inliers.size());
        }
    }
#endif
    const char* envDetails = std::getenv("RSFM_PRINT_MODEL_DETAILS");
    if (envDetails != nullptr && std::stoi(envDetails) == 1) {
        for (const auto& hImg : mFusedImgSeq) {
            o << static_cast<uint32_t>(hImg) << ((&hImg + 1 != mFusedImgSeq.data() + mFusedImgSeq.size()) ? ", " : "\n");
        }
    }
}

void IncreModel::showModel() {
    static const int mode = [](){
        const char* env = std::getenv("SHOW_MODEL");
        if (env == nullptr) {
            return 0;
        }
        return std::stoi(env);
    }();
    if (mode > 0) {
        const auto model = getModel();
        const bool isFirstCall = (mViewer == nullptr);
        if (isFirstCall) {
            mViewer = std::make_unique<DebugModelViewer>();
        }
        mViewer->setModel(model, isFirstCall, cast32u(mFusedImgSeq.size()) - 1u);
    }
    if (mode == 2) {
        std::cout << "Press Enter to Continue";
        std::cin.ignore();
    }
}

void IncreModel::rescaleModel(float scale) {
	for (auto& p : mPoses) {
		p.C = p.C * scale;
		p.v = p.v * scale;
	}
	for (auto& p : mPoints) {
		p.location = p.location * scale;
	}
}

void IncreModel::transform(const Sim3Transform& sim3) {
	const Eigen::Affine3f trans = toAffine3f(sim3);
	auto transform = [&trans](const Vec3f& src) {
		return fromEigen((trans * toEigen(src)).eval());
	};
	const bool rollVGlb = isRollVGlb(mBuilder->config().shutterType);
	for (auto& p : mPoses) {
		p.C = transform(p.C);
		p.R = (p.R * sim3.R.conjugate()).normalized();
		p.v = rollVGlb ? sim3.R * p.v * sim3.scale : p.v * sim3.scale;
	}
	for (auto& p : mPoints) {
		p.location = transform(p.location);
	}
}

} // namespace rsfm
