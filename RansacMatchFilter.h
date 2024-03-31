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
#include <CudaStreamMutex.h>
#include <utility>
#include "Types.h"
#include "StorageFwd.h"
#include "fwd.h"
#include "FiberUtils.h"

namespace rsfm
{
class RansacMatchFilter
{
public:
    RansacMatchFilter(Runtime& runtime) : mRuntime{runtime} {}
    ~RansacMatchFilter();
    // threshold is relative to cell width
    std::vector<bool> getInlierMask(Vec2<int> imgSize0, const std::vector<std::pair<Vec2f, Vec2f>>& matches, uint32_t minVotes = 2, uint32_t nbRansacTests = 256, float threshold = 0.15f, uint32_t cellCols = 16, bool tryOtherAffines = false, cudaStream_t stream = nullptr);
private:
    Runtime& mRuntime;
    using CacheKey = cudapp::storage::CacheObjKeyType;
    cudapp::CudaStreamMutex<fb::mutex> mMutex{}; // protects scratch memory for stream safety and key/capacity for thread-safety

    CacheKey mDevScratchKey = cudapp::storage::kInvalidKey;
    size_t mDevScratchBytes = 0;
    CacheKey mPinnedScratchKey = cudapp::storage::kInvalidKey;
    size_t mPinnedScratchBytes = 0;
};
} // namespace rsfm


