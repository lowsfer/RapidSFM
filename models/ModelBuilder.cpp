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

#include "ModelBuilder.h"
#include "IncreModel.h"
#include "../Builder.h"
#include <FiberUtils.h>
#include <unordered_set>
#include <sstream>
#include <PriorityFiberPool.h>

namespace rsfm
{
ModelBuilder::ModelBuilder(Builder& builder) : mBuilder{&builder} {}
ModelBuilder::~ModelBuilder() = default;

void ModelBuilder::addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> pairs)
{
#if 0
    std::stringstream ss;
    ss << makeFmtStr("image #%u is paired with:", static_cast<uint32_t>(hImage));
    for (const auto& x : pairs) {
        ss << " " << static_cast<uint32_t>(x->images.first);
    }
    ss << '\n';
    printf(ss.str().c_str());
#endif
    if (mModels.empty()) {
        mPendingModel = mHandleGen.make();
        mModels.try_emplace(mPendingModel, std::make_unique<IncreModel>(*mBuilder));
    }
    assert(mPendingModel != kInvalid<ModelHandle>);

    if (mUnusedPairs.count(hImage) != 0) {
        for (auto& p : mUnusedPairs.at(hImage)) {
            pairs.emplace_back(std::move(p));
        }
        mUnusedPairs.erase(hImage);
    }

    std::unordered_map<ModelHandle, std::vector<std::unique_ptr<ImagePair>>> mergeCandidates;
    std::unordered_set<ImageHandle> reAddCandidates;
    for (const auto& p : pairs) {
        const ImageHandle hFirst = p->images.first;
        if (mImgGrouping.has(hFirst)) {
            const ModelHandle hModel = mImgGrouping.at(hFirst);
            if (mModels.at(hModel)->isInitialized()) {
                mergeCandidates.try_emplace(hModel);
            }
            else {
                reAddCandidates.emplace(hFirst);
            }
        }
    }

    std::vector<ModelHandle> modelHandles;
    modelHandles.reserve(mModels.size());
    for (const auto& m : mModels) {
        modelHandles.emplace_back(mModels.getKey(m));
    }
    if (modelHandles.size() > 1)
    {
        VectorMap<ModelHandle, float> affinity;
        for (const ModelHandle h : modelHandles) {
            affinity.try_emplace(h, mModels.at(h)->checkAffinity(hImage, pairs));
        }
        std::sort(modelHandles.begin(), modelHandles.end(), [&](ModelHandle x, ModelHandle y){
            return affinity.at(x) > affinity.at(y);
        });
        modelHandles.erase(std::remove_if(modelHandles.begin(), modelHandles.end(), [&](ModelHandle h){
            return affinity.at(h) < 0;
        }), modelHandles.end());
        ASSERT(!mModels.at(modelHandles.back())->isInitialized());
    }
    bool isAdded = false;
    ModelHandle hModelAddedInto = kInvalid<ModelHandle>;
    for (const ModelHandle hModel : modelHandles) {
        const auto& model = mModels.at(hModel);
        isAdded = model->addImage(hImage, pairs);
        if (!isAdded) {
            model->retriangulate();
            model->bundleAdjustment(IncreModel::BAMethod::kGlobal, false);
            // model->bundleAdjustment(IncreModel::BAMethod::kLocal, false);  // should include paired images and pnpPts. See issue #7
            isAdded = model->addImage(hImage, pairs);
        }
        // if (isAdded && model->isInitialized()) {
        //     printf("addImage #%u %s\n", static_cast<uint32_t>(hImage), isAdded && model->isInitialized() ? "success" : "failure");
        // }
        // printf("Adding image #%u into model # %u: %s\n", (uint32_t)hImage, (uint32_t)hModel, isAdded && model->isInitialized() ? "success" : "failure");
        if (isAdded) {
            hModelAddedInto = hModel;
            break;
        }
    }
    ASSERT(isAdded);
    if (hModelAddedInto != mPendingModel) { // if mPendingModel is initialized AND hImage is added into it, mImgGrouping will be updated later
        ASSERT(mImgGrouping.try_emplace(hImage, hModelAddedInto).second);
    }
    else {
        if (mModels.at(hModelAddedInto)->isInitialized()) { //@fixme: even if mPendingModel is now initialized, hImage is not necessarily added successfully
            auto& lastModel = *mModels.at(mPendingModel);
            for (ImageHandle h: lastModel.getFusedImages())
            {
                ASSERT(mImgGrouping.try_emplace(h, mPendingModel).second);
            }
            for (const auto& h : lastModel.getPending()) {
                ASSERT(!mImgGrouping.has(h));
            }
        }
    }
    assert(pairs.empty());
    if (mModels.at(hModelAddedInto)->isInitialized()) { //@fixme: even if mPendingModel is now initialized, hImage is not necessarily added successfully
        std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>> tmp;
        auto& model = *mModels.at(hModelAddedInto);
        model.collectUnusedPairs(tmp);
        for (auto& [hSecond, pairs] : tmp) {
            for (auto& p : pairs) {
                unused(p);
                assert(p->images.second == hSecond && model.hasFused(p->images.first) && !model.hasFused(hSecond) &&
                    (!mImgGrouping.has(hSecond) || hModelAddedInto != mImgGrouping.at(p->images.second)));
            }
            const bool isLinkPair = mImgGrouping.has(hSecond);
            auto& dst = isLinkPair ? mLinkPairs : mUnusedPairs[hSecond];
            for (auto& p : pairs) {
                dst.emplace_back(std::move(p));
            }
            if (!isLinkPair) {
                reAddCandidates.emplace(hSecond);
            }
        }
        model.showModel();
    }

    std::vector<std::pair<ImageHandle, std::vector<std::unique_ptr<ImagePair>>>> pendingPairs;
    if (mModels.at(mPendingModel)->isInitialized()) { //@fixme: even if mPendingModel is now initialized, hImage is not necessarily added successfully
        auto& lastModel = *mModels.at(mPendingModel);
        pendingPairs = lastModel.takeAllPendingPairs();
        mPendingModel = mHandleGen.make();
        mModels.try_emplace(mPendingModel, std::make_unique<IncreModel>(*mBuilder));
        if (mImgGrouping.find(hImage) == mImgGrouping.end()) {
            hModelAddedInto = mPendingModel;
        }
        reAddCandidates.clear();
    }
    else if (!reAddCandidates.empty()){
        auto& lastModel = *mModels.at(mPendingModel);
        for (const ImageHandle hCandidate : reAddCandidates) {
            if (lastModel.isPending(hCandidate)) {
                pendingPairs.emplace_back(hCandidate, lastModel.takePendingPairs(hCandidate));
                ASSERT(!mImgGrouping.has(hCandidate));
            }
        }
        reAddCandidates.clear(); // @fixme: should we do this? If we do this, "if (!reAddCandidates.empty()) {" becomes dead code
    }

    mergeCandidates.erase(hModelAddedInto);
    if (mModels.at(hModelAddedInto)->isInitialized() && !mergeCandidates.empty())
    {
        ASSERT(mImgGrouping.at(hImage) == hModelAddedInto);
        for (auto& p : mLinkPairs) {
            ASSERT(mImgGrouping.has(p->images.first) && mImgGrouping.has(p->images.second));
            const ModelHandle hModel1 = mImgGrouping.at(p->images.first);
            const ModelHandle hModel2 = mImgGrouping.at(p->images.second);
            ASSERT(hModel1 != hModel2);
            if ((hModel1 == hModelAddedInto && mergeCandidates.count(hModel2)) || (hModel2 == hModelAddedInto && mergeCandidates.count(hModel1))) {
                mergeCandidates.at(hModel1 == hModelAddedInto ? hModel2 : hModel1).emplace_back(std::move(p));
                p.reset();
            }
        }
        
        mLinkPairs.erase(std::remove_if(mLinkPairs.begin(), mLinkPairs.end(), [](auto& x){return x == nullptr;}), mLinkPairs.end());
        
        ModelHandle mergeSrc = hModelAddedInto;
        const auto bestMergeCand = std::max_element(mergeCandidates.begin(), mergeCandidates.end(), [](const auto& x, const auto& y){return x.second.size() < y.second.size();});
        ModelHandle mergeDst = bestMergeCand->first;
        if (mModels.at(mergeDst)->getNbPoints() < mModels.at(mergeSrc)->getNbPoints()) {
            std::swap(mergeDst, mergeSrc);
        }
        for (const auto& p : bestMergeCand->second) {
            if (mImgGrouping.at(p->images.first) != mergeSrc){
                *p = p->inverse();
            }
            ASSERT(mImgGrouping.at(p->images.first) == mergeSrc && mImgGrouping.at(p->images.second) == mergeDst);
        }
        const bool isMerged = mModels.at(mergeDst)->tryMergeModel(*mModels.at(mergeSrc), bestMergeCand->second);
        if (isMerged) {
            // printf("Merging model #%u into #%u\n", (uint32_t)mergeSrc, (uint32_t)mergeDst);
            mModels.erase(mergeSrc);
            for (auto& x : mImgGrouping) {
                if (x == mergeSrc) {
                    x = mergeDst;
                }
            }
            mModels.at(mergeDst)->showModel();
        }
        else {
            for (auto& p : bestMergeCand->second) {
                mLinkPairs.emplace_back(std::move(p));
            }
        }
    }

    // if hModelAddedInto is the old mPendingModel and mPendingModel is updated, reAddCandidates is already cleared
    if (!reAddCandidates.empty()) { // dead code
        assert(!mModels.at(mPendingModel)->isInitialized());
        for (const auto hCandidate : reAddCandidates) {
            ASSERT(!mImgGrouping.has(hCandidate) || mImgGrouping.at(hCandidate) == mPendingModel);
            auto pairs = mModels.at(mPendingModel)->takePendingPairs(hCandidate);
            ASSERT(!mImgGrouping.has(hCandidate));
            addImage(hCandidate, std::move(pairs));
        }
    }

    for (auto& x : pendingPairs) {
        this->addImage(x.first, std::move(x.second));
    }
}

void ModelBuilder::finish() {
    // Add pending pairs of the pending model again
    {
        assert(!mModels.at(mPendingModel)->isInitialized());
        auto allPendingPairs = mModels.at(mPendingModel)->takeAllPendingPairs();
        for (auto& p : allPendingPairs) { mImgGrouping.erase(p.first); }
        for (auto& pairs : allPendingPairs) {
            addImage(pairs.first, std::move(pairs.second));
        }
    }
    {
        const auto allPendingPairs = mModels.at(mPendingModel)->takeAllPendingPairs();
        if (!allPendingPairs.empty()) {
            printf("Some images are still pending:\n");
            for (const auto& pairs : allPendingPairs) {
                printf("\t%u:  ", static_cast<uint32_t>(pairs.first));
                for (const auto& p : pairs.second) {
                    printf("%u, ", static_cast<uint32_t>(p->images.first));
                }
                printf("\n");
            }
        }
        if (!mUnusedPairs.empty()) {
            printf("Some images are still unused:\n");
            for (const auto& pairs : mUnusedPairs) {
                printf("\t%u:  ", static_cast<uint32_t>(pairs.first));
                for (const auto& p : pairs.second) {
                    printf("%u, ", static_cast<uint32_t>(p->images.first));
                }
                printf("\n");
            }
        }
        if (!mLinkPairs.empty()) {
            printf("Some link pairs are unused:\n");
            for (const auto& p : mLinkPairs) {
                printf("{%u, %u},", static_cast<uint32_t>(p->images.first), static_cast<uint32_t>(p->images.second));
            }
            printf("\n");
        }
    }
    // @todo: Try merge models
    

    int idxModel = 0;
    for (const auto& m : mModels) {
        if (m->isInitialized()) {
            printf("Finishing model #%d\n", idxModel);
            m->finish();
            m->printSummary(std::cout);
            idxModel++;
        }
    }
}

const IncreModel* ModelBuilder::getModel(size_t idx) const
{
    auto iter = mModels.begin();
    for (size_t i = 0; i < idx; i++) {iter++;}
    return (*iter).get();
}

void ModelBuilder::writePly(const std::string& filenamePattern) const
{
    std::vector<fb::future<void>> futures;
    int i = 0;
    for (const auto& model : mModels) {
        if (model->isInitialized()) {
            futures.emplace_back(mBuilder->fiberPool()->post([&model, filename{makeFmtStr(filenamePattern.c_str(), i++)}](){
                model->writePly(filename.c_str());
            }));
        }
    }
    for (auto& f : futures) {f.get();}
}

void ModelBuilder::writeNvm(const std::string& filenamePattern) const
{
    std::vector<fb::future<void>> futures;
    int i = 0;
    for (const auto& model : mModels) {
        if (model->isInitialized()) {
            futures.emplace_back(mBuilder->fiberPool()->post([&model, filename{makeFmtStr(filenamePattern.c_str(), i++)}](){
                model->writeNvm(filename.c_str());
            }));
        }
    }
    for (auto& f : futures) {f.get();}
}

void ModelBuilder::writeRsm(const std::string& filenamePattern) const
{
    std::vector<fb::future<void>> futures;
    int i = 0;
    for (const auto& model : mModels) {
        if (model->isInitialized()) {
            futures.emplace_back(mBuilder->fiberPool()->post([&model, filename{makeFmtStr(filenamePattern.c_str(), i++)}](){
                model->writeRsm(filename.c_str());
            }));
        }
    }
    for (auto& f : futures) {f.get();}
}

void ModelBuilder::debugCheckBadUnusedPairs() const
{
    for (const auto& [hImage, pairs] : mUnusedPairs)
    {
        ASSERT(!mImgGrouping.has(hImage));
        for (const auto& p : pairs) {
            ASSERT(hImage == p->images.second);
            ASSERT(mImgGrouping.has(p->images.first));
        }        
    }
}

void ModelBuilder::debugCheckImgGrouping() const
{
    for (auto iter = mModels.begin(); iter != mModels.end(); iter++) {
        const auto& m = *iter;
        for (const auto& h : m->getFusedImages()) {
            ASSERT(mImgGrouping.has(h) && mImgGrouping.at(h) == iter.key());
        }
        for (const auto& h : m->getPending()) {
            ASSERT(!mImgGrouping.has(h));
        }
    }
    for (auto iter = mImgGrouping.begin(); iter != mImgGrouping.end(); iter++) {
        ASSERT(mModels.at(*iter)->hasFused(iter.key()));
    }
}

void ModelBuilder::debugCheckLinkPairs() const
{
    for (const auto& p : mLinkPairs) {
        const auto [hFirst, hSecond] = p->images;
        ASSERT(mImgGrouping.has(hFirst) && mImgGrouping.has(hSecond));
        ASSERT(mImgGrouping.at(hFirst) != mImgGrouping.at(hSecond));
    }
}

void ModelBuilder::debugSanityCheck() const
{
    debugCheckBadUnusedPairs();
    debugCheckImgGrouping();
    debugCheckLinkPairs();
}
}// namespace rsfm

