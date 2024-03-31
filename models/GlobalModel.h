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
#include "../Types.h"
#include <boost/container/flat_map.hpp>

namespace rsfm
{
class GlobalModel : public DefaultModel
{
public:
    bool canFuse(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs);
    // before addImage call, hImage must be checked to confirm that it belongs to this model.
    void addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> allPairs) override;
protected:
    void groupObserv(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs);
private:
    bool tryAddImage(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs);
    bool tryInitializeModel();
private:
    using IdxObPropMap = boost::typed_identity_property_map<IdxOb>;
    boost::disjoint_sets_with_storage<IdxObPropMap, IdxObPropMap> mObGroups;
    std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>, DirectMappingHash<ImageHandle>> mPendingPairs;
    std::vector<ImageHandle> mPendingSeq;
    std::unordered_map<ImageHandle, Pose, DirectMappingHash<ImageHandle>> mPoses;
    // @todo: also store Transform + t-ratio for pose graph optimization
};
} // namespace rsfm


namespace rsfm
{
void GlobalModel::addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> allPairs)
{
    assert(canFuse(hImage, allPairs));
    assert(std::all_of(allPairs.begin(), allPairs.end(), [hImage](const std::unique_ptr<ImagePair>& p){return hImage == p->images.second;}));
    mPendingPairs.emplace(hImage, std::move(allPairs));
    mPendingSeq.emplace_back(hImage);
    if (mPoses.empty()) {
        const bool initSuccess = tryInitializeModel();
        if (!initSuccess){
            return;
        }
    }
    ASSERT(mPoses.size() >= 3);
    const auto isInvalid = [](const ImageHandle& h){ return h == kInvalid<ImageHandle>; };
    do {
        mPendingSeq.erase(std::remove_if(mPendingSeq.begin(), mPendingSeq.end(), isInvalid), mPendingSeq.end());
        assert(mPendingSeq.size() == mPendingPairs.size());
        for (ImageHandle& pImg : mPendingSeq){
            const auto pendingPairs = mPendingPairs.at(pImg);
            const bool chooseSuccess = tryAddImage(pImg, pendingPairs);
            if (chooseSuccess) {
                groupObserv(pImg, pendingPairs);
                pImg = kInvalid<ImageHandle>;
                mPendingPairs.erase(pImg);
            }
        }
    }while(std::any_of(mPendingSeq.begin(), mPendingSeq.end(), isInvalid));    
}
} // namespace rsfm


#include "../Types.hpp"
#include "../Builder.h"
namespace rsfm{
void GlobalModel::groupObserv(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& allPairs)
{
    ASSERT(std::all_of(allPairs.begin(), allPairs.end(), [hImage](const std::unique_ptr<ImagePair>& x){
        return x->images.second == hImage;}));
    ASSERT(mIdxObOffset.try_emplace(hImage, mNextIdxObOffset).second);
    mImgSeqWIdxOb.push_back(hImage);
	const auto& img = *mBuilder->getImage(hImage);
	makeIdxObOffsetForImage(hImage, img.nbKPoints, img.getNbTiePtMeasurements());

    for (const auto& imgPair : allPairs) {
        const ImageHandle hFirst = imgPair->images.first;
        const ImageHandle hSecond = imgPair->images.second;
        assert(hSecond == hImage);
        const IdxOb idxObOffsetFirst = mIdxObOffset.at(hFirst);
        const IdxOb idxObOffsetSecond = mIdxObOffset.at(hSecond);
        std::vector<bool>& obMaskFirst = mObMask.at(hFirst);
        std::vector<bool>& obMaskSecond = mObMask.at(hSecond);
        ASSERT(imgPair->solutions.size() == 1u && "Solution not decided");
        const auto& solution = imgPair->solutions.at(0);
        for (auto& idxInlier : solution.inliers){
            const auto& m = imgPair->kptsMatches.at(idxInlier);
            mObGroups.union_set(idxObOffsetFirst + m.first, idxObOffsetSecond + m.second);
            obMaskFirst.at(m.first) = true;
            obMaskSecond.at(m.second) = true;
        }
    }
}

//@fixme: add test
bool GlobalModel::tryInitializeModel()
{
    ASSERT(mPoses.size() == 0);
    if (mPendingSeq.size() < 3){
        return false;
    }
    const ImageHandle hImage = mPendingSeq.back();
    const auto& pairs = mPendingPairs.at(hImage);
    if (pairs.size() < 2) {
        return false;
    }

    for (uint32_t idxP0 = 0; idxP0 < pairs.size(); idxP0++) {
        for (uint32_t idxP1 = 0; idxP1 < pairs.size(); idxP1++) {
            const ImagePair& p0 = *pairs.at(idxP0);
            const ImagePair& p1 = *pairs.at(idxP1);
            assert(p0.images.second == hImage && p1.images.second == hImage);
            const ImageHandle hFirst = std::min(p0.images.first, p1.images.first);
            const ImageHandle hSecond = std::max(p0.images.first, p1.images.first);
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
            const bool isP2Fwd = (hFirst == p0.images.first); // p0.images.first < p1.images.first;
            assert(p2.images.second == hSecond);
            std::vector<std::pair<std::array<uint32_t, 3>, float>> solutionConsistencyErrors;
            solutionConsistencyErrors.reserve(p0.solutions.size() * p1.solutions.size() * p2.solutions.size());
            for (uint32_t i0 = 0; i0 < p0.solutions.size(); i0++){
                for (uint32_t i1 = 0; i1 < p1.solutions.size(); i1++){
                    for (uint32_t i2 = 0; i2 < p2.solutions.size(); i2++){
                        const Rotation rotErr = (isP2Fwd ? p2.solutions[i2].transform.R.inverse() : p2.solutions[i2].transform.R) * p1.solutions[i1].transform.R.inverse() * p0.solutions[i0].transform.R;
                        const float angleErr = Eigen::AngleAxisf{toEigen(rotErr)}.angle();
                        solutionConsistencyErrors.emplace_back(std::array<uint32_t, 3>{{i0, i1, i2}}, angleErr);
                    }
                }
            }
            if (!solutionConsistencyErrors.empty()) {
                const auto iterMinErr = std::min_element(solutionConsistencyErrors.begin(), solutionConsistencyErrors.end(), [](const auto& a, const auto& b){return a.second < b.second;});
                const auto& idxS = iterMinErr->first;
                mPoses.try_emplace(hImage, Pose{Rotation::identity(), {0.f,0.f,0.f}});
                const auto& trans0 = p0.solutions.at(idxS[0]).transform;
                const auto& trans1 = p1.solutions.at(idxS[1]).transform;
                const auto& trans2 = p2.solutions.at(idxS[2]).transform;
                Rotation r0, r1;
                Eigen::Vector4f::Map(r0.data()) = (1.f/3 * (Eigen::Vector4f::Map((trans0.R.inverse() * mPoses.at(hImage).R).data()) * 2.f +
                    Eigen::Vector4f::Map(((isP2Fwd ? trans2.R.inverse() : trans2.R) * trans1.R.inverse() * mPoses.at(hImage).R).data()))).normalized();
                Eigen::Vector4f::Map(r1.data()) = (1.f/3 * (Eigen::Vector4f::Map((trans1.R.inverse() * mPoses.at(hImage).R).data()) * 2.f +
                    Eigen::Vector4f::Map(((isP2Fwd ? trans2.R : trans2.R.inverse()) * trans0.R.inverse() * mPoses.at(hImage).R).data()))).normalized();
                const Vec3f c0 = r0.inverse() * trans0.R.inverse() * (mPoses.at(hImage).R * mPoses.at(hImage).C + trans0.t);
                const Vec3f c1 = scale * (r1.inverse() * trans1.R.inverse() * (mPoses.at(hImage).R * mPoses.at(hImage).C + trans1.t));
                mPoses.try_emplace(p0.images.first, Pose{r0, c0});
                mPoses.try_emplace(p1.images.first, Pose{r1, c1});
                // @fixme: run bundle adjustment
                return true;
            }
        }
    }
    return false;
}
} // namespace rsfm
