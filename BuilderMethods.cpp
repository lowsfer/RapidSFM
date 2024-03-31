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
#include <Runtime.hpp>
#include "Config.h"
#include "Builder.h"
#include <algorithm>
#include <RapidSift.h>
#include <vector>
#include <cuda_utils.h>
#include <StorageManager.h>
#include <DefaultCacheableObject.h>
#include "Image.h"
#include "ImagePair.h"
#include "Types.hpp"
#include <FiberUtils.h>
#include "legacy/PropagationMatchFilter.h"
#include <boost/fiber/cuda/waitfor.hpp>
#include <FiberUtils.h>
#include <functional>
#include <algorithm>
#include "models/ModelBuilder.h"
#include "bruteForceMatcher.h"
#include <PriorityFiberPool.h>
#include "RansacMatchFilter.h"
#include "models/rbaUtils.h"

using cudapp::storage::StorageLocation;
using cudapp::storage::DefaultCacheableObject;
using cudapp::storage::AcquiredObj;
template <typename T>
using AcquiredMemory = cudapp::storage::AcquiredMemory<T>;

namespace rsfm {

void Builder::concatDesc(ImageHandle hImage) {
    const cudaStream_t stream = anyStream();
    const auto img = getImage(hImage);
    const AcquiredMemory<const SiftDescriptor> descHolder = cudapp::storage::acquireMemory<const SiftDescriptor>(storageManager(), img->descriptors, StorageLocation::kSysMem, stream, false, true);
    cudapp::fiberSyncCudaStream(stream);
    assert(descHolder.nbElems() == img->nbKPoints);
    static_assert(sizeof(mAllDesc[0]) == sizeof(descHolder.data()[0]));
    mAllDesc.insert(mAllDesc.end(),
        reinterpret_cast<const std::array<uint8_t, 128>*>(descHolder.data()),
        reinterpret_cast<const std::array<uint8_t, 128>*>(descHolder.data()) + img->nbKPoints);
}

void Builder::buildVocabulary(uint32_t nbImages) {
    ASSERT(mVocabulary == nullptr);
    const auto stream = anyStream();
    const auto nbDesc = mAllDesc.size();
    const auto devDesc = cudaMemPool<CudaMemType::kDevice>().alloc<std::array<uint8_t, 128>>(nbDesc, stream);
    cudaCheck(cudaMemcpyAsync(devDesc.get(), mAllDesc.data(), sizeof(mAllDesc[0]) * nbDesc, cudaMemcpyHostToDevice, stream));
    const uint32_t branchFactor = 16u;
    const uint32_t nbDoc = nbImages;
    uint32_t nbLevels = 0u;
    while (std::pow(branchFactor, nbLevels) * 8u < 10000u * nbDoc) {
        nbLevels++;
    }
    mVocabulary = rbow::buildVocabulary(rbow::SiftAttr, 16, devDesc.get(), nbDesc, nbDoc, nbLevels, stream);
    mDatabase = rbow::createDataBase(mVocabulary.get());
    if (config().vocabulary.saveAfterRebuild)
    {
        const auto serializedVoc = mVocabulary->serialize();
        const auto& vocPath = config().getVocabularyPath();
        if (fs::exists(vocPath)) {
            fs::remove(vocPath);
        }
        auto const vocDir = fs::path{vocPath}.parent_path();
        if (!fs::exists(vocDir)) {
            if (!fs::create_directories(vocDir)) {
                fprintf(stderr, "Error: failed to create directory %s\n", vocDir.c_str());
                throw std::runtime_error("Failed to create directory");
            }
        }
        else if (!fs::is_directory(vocDir)) {
            fprintf(stderr, "Error: %s is not a directory\n", vocDir.c_str());
            throw std::runtime_error("vocabulary parent path is not a directory");
        }
        saveBinaryFile(vocPath, serializedVoc.data(), serializedVoc.size());
    }
    cudapp::fiberSyncCudaStream(stream);
    mAllDesc.clear();
}

std::vector<std::unique_ptr<Matches>> Builder::matchImages(std::vector<std::pair<ImageHandle, std::vector<Pair<Index>>>> hFirstCandidates, ImageHandle hSecond)
{
    if (hFirstCandidates.empty()) {
        return {};
    }
    const Image* imgSecond = nullptr;
    std::vector<const Image*> imgFirst(hFirstCandidates.size(), nullptr);
    {
        std::shared_lock<std::shared_mutex> lk{mLock};
        imgSecond = mImages.at(hSecond).get();
        std::transform(hFirstCandidates.begin(), hFirstCandidates.end(), imgFirst.begin(), [this](const auto& x){
			return mImages.at(x.first).get();});
    }
    const cudaStream_t stream = anyStream();
    const bool crossCheck = config().pair.crossCheck;
    const auto matchesOrig = [&]{
        const AcquiredMemory<const SiftDescriptor> descHolderSecond = cudapp::storage::acquireMemory<const SiftDescriptor>(storageManager(), imgSecond->descriptors, StorageLocation::kCudaDeviceMem, stream, false, true);
        std::vector<AcquiredMemory<const SiftDescriptor>> descHolderFirst(hFirstCandidates.size());
        std::transform(imgFirst.begin(), imgFirst.end(), descHolderFirst.begin(), [this, stream](const Image* img){
            return cudapp::storage::acquireMemory<const SiftDescriptor>(storageManager(), img->descriptors, StorageLocation::kCudaDeviceMem, stream, false, true);
        });
        const SiftDescriptor* const descSecond = descHolderSecond.data();
        std::vector<const SiftDescriptor*> descFirst(hFirstCandidates.size());
        std::transform(descHolderFirst.begin(), descHolderFirst.end(), descFirst.begin(), [](const AcquiredMemory<const SiftDescriptor>& holder){
            return holder.data();
        });
		auto makeMatcherImg = [](const Image& img, const SiftDescriptor* desc) {
            using Desc = BruteForceMatcher::Descriptor;
            static_assert(sizeof(Desc) == sizeof(SiftDescriptor));
            return typename BruteForceMatcher::Img{
                reinterpret_cast<const Desc*>(desc),
                img.nbKPoints,
                img.hImage
            };
		};
        std::vector<typename BruteForceMatcher::Img> queries(imgFirst.size());
        for (size_t i = 0; i < imgFirst.size(); i++) {
			queries.at(i) = makeMatcherImg(*imgFirst.at(i), descFirst.at(i));
        }
		auto train = makeMatcherImg(*imgSecond, descSecond);
        return mMatcher->match(queries, train, crossCheck, stream);
    }().get();

    // filter by cross check and PMF
    std::vector<fb::future<std::vector<std::pair<Index, Index>>>> allFutureMatches(matchesOrig.size());
    for (uint32_t i = 0; i < hFirstCandidates.size(); i++) {
        auto filterMatches = [this, crossCheck, stream](
                const Image* first, const std::vector<typename BruteForceMatcher::BestMatch>* queryMatches,
                const Image* second, const std::vector<typename BruteForceMatcher::BestMatch>* trainMatches)
        {
            std::vector<ValidMatch> validMatches;
            assert(queryMatches->size() == first->nbKPoints);
            if(crossCheck) {
                assert(trainMatches->size() == second->nbKPoints);
                validMatches = crossCheckMatches(queryMatches->data(), first->nbKPoints, trainMatches->data(), second->nbKPoints);
            }
            else{
                assert(trainMatches->empty());
                validMatches = removeMatchConflicts(queryMatches->data(), first->nbKPoints, second->nbKPoints);
            }
            std::vector<std::pair<Vec2f, Vec2f>> matchPts(validMatches.size());
            {
                const StorageLocation kptsLocation = StorageLocation::kSysMem;
                const auto kptsFirstHolder = cudapp::storage::acquireMemory<const KeyPoint>(storageManager(), first->keyPoints, kptsLocation, stream, false, true);
                const auto kptsSecondHolder = cudapp::storage::acquireMemory<const KeyPoint>(storageManager(), second->keyPoints, kptsLocation, stream, false, true);
                cudapp::fiberSyncCudaStream(stream);

                assert(kptsFirstHolder.obj()->getCurrentStorageBytes(kptsLocation) == sizeof(KeyPoint) * first->nbKPoints);
                assert(kptsSecondHolder.obj()->getCurrentStorageBytes(kptsLocation) == sizeof(KeyPoint) * second->nbKPoints);
                const KeyPoint* kptsFirst = kptsFirstHolder.data();
                const KeyPoint* kptsSecond = kptsSecondHolder.data();
                assert(kptsFirst != nullptr);
                assert(kptsSecond != nullptr);
                std::transform(validMatches.begin(), validMatches.end(), matchPts.begin(), [kptsFirst, kptsSecond](const ValidMatch& x){
                    return std::make_pair(fromRBA(kptsFirst[x.idxQuery].location), fromRBA(kptsSecond[x.idxTrain].location));
                });
            }

            std::vector<bool> inlierMask;
            if (config().matchFilter.useGPU) {
                auto& filterCfg = config().matchFilter;
                inlierMask = mMatchFilters.get()->getInlierMask({first->width, first->height}, matchPts, filterCfg.minVotes, filterCfg.nbRansacTests, filterCfg.relativeThreshold, filterCfg.cellCols, filterCfg.tryOtherAffines, stream);
            }
            else {
                pmf::PropagationMatchFilter pmfFilter{{first->width, first->height}, matchPts};
                inlierMask = pmfFilter.getInlierMask(cast32i(config().pair.pmfMinVotes));
            }

            assert(inlierMask.size() == validMatches.size());
            std::vector<std::pair<Index, Index>> matches;
            matches.reserve(inlierMask.size());
            for (uint32_t j = 0; j < inlierMask.size(); j++) {
                if (inlierMask[j]) {
                    const auto& m = validMatches[j];
                    matches.emplace_back(std::pair<Index, Index>{m.idxQuery, m.idxTrain});
                }
            }
            matches.shrink_to_fit();
            return matches;
        };
        allFutureMatches.at(i) = fiberPool()->post(filterMatches, imgFirst.at(i), &matchesOrig.at(i).first, imgSecond, &matchesOrig.at(i).second);
    }
    std::vector<std::unique_ptr<Matches>> allMatches(hFirstCandidates.size());
    for (uint32_t i = 0; i < hFirstCandidates.size(); i++) {
		auto& cand = hFirstCandidates.at(i);
        allMatches[i] = std::make_unique<Matches>(Matches{
			{cand.first, hSecond},
			allFutureMatches.at(i).get(),
			std::move(cand.second)
		});
    }

    notifyProgress();
    return allMatches;
}

std::vector<std::unique_ptr<ImagePair>> Builder::solveImagePairs(ImageHandle hSecond, std::vector<std::unique_ptr<Matches>> allMatches)
{
    unused(hSecond);
    assert(std::all_of(allMatches.begin(), allMatches.end(), [hSecond](const auto& x){return x->images.second == hSecond;}));
    std::vector<fb::future<ImagePair>> futurePairs;
    futurePairs.reserve(allMatches.size());
    for (const auto& m : allMatches) {
        auto asyncCreateImgPair = [this, &m](){ return createImgPair(this, m->images.first, m->images.second, std::move(m->kptsMatches), std::move(m->tiePtMatches)); };
        futurePairs.emplace_back(fiberPool()->post(asyncCreateImgPair));
    }

    std::vector<std::unique_ptr<ImagePair>> result;
    result.reserve(futurePairs.size());
    for (auto& f : futurePairs) {
        ImagePair imgPair = f.get();
        if (!imgPair.solutions.empty()) {
            result.emplace_back(std::make_unique<ImagePair>(std::move(imgPair)));
        }
    }
    result.shrink_to_fit();

    static const bool isVerbose = std::getenv("RSFM_VERBOSE") && std::stoi(std::getenv("RSFM_VERBOSE")) != 0;
    if (isVerbose) {
        std::unordered_set<ImageHandle> solved;
        for (const auto& p : result) {
            solved.emplace(p->images.first);
        }
        std::stringstream ss;
        ss << "solveImagePairs for " << static_cast<uint32_t>(hSecond) << ":";
        for (const auto& m : allMatches) {
            const ImageHandle hFirst = m->images.first;
            const int32_t colorCode = solved.find(hFirst) == solved.end() ? 31 : 32;
            ss << " \033[" << colorCode << "m" << static_cast<uint32_t>(hFirst) << "\033[0m";
        }
        const auto line = ss.str();
        printf("%s\n", line.c_str());
    }

    notifyProgress();
    return result;
}

void Builder::addImageToModels(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> pairs)
{
    mModelBuilder->addImage(hImage, std::move(pairs));
    notifyProgress();
}

} // namespace rsfm
