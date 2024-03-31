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
#include <memory>
#include "../fwd.h"
#include "../RapidSFM.h"
#include <VectorMap.h>
#include "../HandleGenerator.h"
namespace rsfm
{
class ModelBuilder
{
public:
    ModelBuilder(Builder& builder);
    void addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>> pairs);
    void finish();
    size_t getNbModels() const {return mModels.size() - 1u; }
    const IncreModel* getModel(size_t idx) const;
    
    void writePly(const std::string& filenamePattern = "cloud_%u.ply") const;
    void writeNvm(const std::string& filenamePattern = "cloud_%u.nvm") const;
    void writeRsm(const std::string& filenamePattern = "cloud_%u.rsm") const;

    ~ModelBuilder();
private:
    void debugCheckBadUnusedPairs() const;
    void debugCheckImgGrouping() const;
    void debugCheckLinkPairs() const;
    void debugSanityCheck() const;
private:
    Builder* mBuilder;
    enum class ModelHandle : uint32_t{};
    HandleGenerator<ModelHandle> mHandleGen;
    VectorMap<ImageHandle, ModelHandle> mImgGrouping; // only contain images added in successfully initialized model. pending images are not included.
    VectorMap<ModelHandle, std::unique_ptr<IncreModel>> mModels;
    ModelHandle mPendingModel = kInvalid<ModelHandle>; // uninitialized model
    // pairs with hSecond not added to any model but hFirst was. key is hSecond
    std::unordered_map<ImageHandle, std::vector<std::unique_ptr<ImagePair>>> mUnusedPairs;
    // pairs with two images added to two different models
    std::vector<std::unique_ptr<ImagePair>> mLinkPairs;
};

} // namespace rsfm

