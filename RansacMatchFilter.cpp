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

#include "RansacMatchFilter.h"
#include "RansacMatchFilter.cuh"
#include "Runtime.hpp"
#include "FiberUtils.h"

namespace rsfm
{
RansacMatchFilter::~RansacMatchFilter() {
    const auto stream = mRuntime.anyStream();
    auto lock = mMutex.acquire(stream);
    if (mDevScratchKey != Runtime::kInvalidKey) {
        assert(mRuntime.storageManager().hasItem(mDevScratchKey));
        mRuntime.storageManager().removeItem(mDevScratchKey);
        mDevScratchKey = Runtime::kInvalidKey;
    }
    if (mPinnedScratchKey != Runtime::kInvalidKey) {
        assert(mRuntime.storageManager().hasItem(mPinnedScratchKey));
        mRuntime.storageManager().removeItem(mPinnedScratchKey);
        mPinnedScratchKey = Runtime::kInvalidKey;
    }
}

std::vector<bool> RansacMatchFilter::getInlierMask(Vec2<int> imgSize0, const std::vector<std::pair<Vec2f, Vec2f>>& matches, uint32_t minVotes, uint32_t nbRansacTests, float threshold, uint32_t cellCols, bool tryOtherAffines, cudaStream_t stream)
{
    const uint32_t cellWidth = divUp(std::max(imgSize0.x, imgSize0.y), cellCols);
    const uint32_t cols = divUp(imgSize0.x, cellWidth);
    const uint32_t rows = divUp(imgSize0.y, cellWidth);
    const uint32_t nbCells = cols * rows;
    ASSERT(cols >= 1 && rows >= 1);

    std::vector<std::vector<uint32_t>> cellIndices(cols*rows);
    for (uint32_t n = 0; n < matches.size(); n++) {
        const auto& [left, right] = matches.at(n);
        const auto i = static_cast<uint32_t>(clamp(std::ceil(left.y / cellWidth), 0.f, rows - 1.f));
        const auto j = static_cast<uint32_t>(clamp(std::ceil(left.x / cellWidth), 0.f, cols - 1.f));
        const uint32_t idxCell = i * cols + j;
        cellIndices.at(idxCell).push_back(n);
    }
    using namespace rmf;
    std::vector<uint32_t> flatCellIndices;
    std::vector<PtPair> reorderedMatches; reorderedMatches.reserve(matches.size());
    std::vector<uint32_t> bounds({0}); bounds.reserve(nbCells + 1);
    for (const auto& cell : cellIndices) {
        for (const auto& idx : cell) {
            const auto& [l, r] = matches.at(idx);
            reorderedMatches.emplace_back(PtPair{{l.x, l.y}, {r.x, r.y}});
            flatCellIndices.emplace_back(idx);
        }
        bounds.push_back(bounds.back() + cast32u(cell.size()));
    }

    const size_t alignment = 16;
    const size_t voteBytes = roundUp(sizeof(uint8_t) * matches.size(), alignment);
    const size_t affineBytes = roundUp(sizeof(kmat<float, 2, 3>) * nbCells, alignment);
    const size_t matchBytes = roundUp(sizeof(rmf::PtPair) * matches.size(), alignment);
    const size_t boundBytes = roundUp(sizeof(uint32_t) * (nbCells + 1), alignment);

    const size_t devScratchBytes = voteBytes + affineBytes + matchBytes + boundBytes;
    const size_t pinnedScratchBytes = std::max(matchBytes + boundBytes, voteBytes);

    std::vector<uint8_t> votes(matches.size());
    {
        auto lock = mMutex.acquire(stream);
        if (mDevScratchBytes < devScratchBytes) {
            if (mDevScratchKey != Runtime::kInvalidKey) {
                assert(mRuntime.storageManager().hasItem(mDevScratchKey));
                mRuntime.storageManager().removeItem(mDevScratchKey);
                mDevScratchKey = Runtime::kInvalidKey;
            }
            mDevScratchKey = mRuntime.allocCacheableScratch<std_byte, CudaMemType::kDevice>(devScratchBytes);
            mDevScratchBytes = devScratchBytes;
        }
        if (mPinnedScratchBytes < pinnedScratchBytes) {
            if (mPinnedScratchKey != Runtime::kInvalidKey) {
                assert(mRuntime.storageManager().hasItem(mPinnedScratchKey));
                mRuntime.storageManager().removeItem(mPinnedScratchKey);
                mPinnedScratchKey = Runtime::kInvalidKey;
            }
            mPinnedScratchKey = mRuntime.allocCacheableScratch<std_byte, CudaMemType::kPinned>(pinnedScratchBytes);
            mPinnedScratchBytes = pinnedScratchBytes;
        }
        const auto devScratchHolder = cudapp::storage::acquireScratch<std_byte, CudaMemType::kDevice>(mRuntime.storageManager(), mDevScratchKey, stream, true, true);
        const auto devScratch = devScratchHolder.data();
        const auto pVotes = reinterpret_cast<uint8_t*>(devScratch);
        const auto pAffine = reinterpret_cast<kmat<float, 2, 3>*>(devScratch + voteBytes);
        const auto pMatches = reinterpret_cast<rmf::PtPair*>(devScratch + voteBytes + affineBytes);
        const auto pBounds = reinterpret_cast<uint32_t*>(devScratch + voteBytes + affineBytes + matchBytes);

        const auto pinnedScratchHolder = cudapp::storage::acquireScratch<std_byte, CudaMemType::kPinned>(mRuntime.storageManager(), mPinnedScratchKey, stream, true, true);
        const auto pinnedScratch = pinnedScratchHolder.data();
        const auto pPinnedMatches = reinterpret_cast<rmf::PtPair*>(pinnedScratch);
        const auto pPinnedBounds = reinterpret_cast<uint32_t*>(pinnedScratch + matchBytes);
        const auto pPinnedVotes = reinterpret_cast<uint8_t*>(pinnedScratch);

        assert(matches.size() == reorderedMatches.size());
        launchCudaHostFunc(stream, [reorderedMatches{std::move(reorderedMatches)}, bounds{std::move(bounds)}, pPinnedMatches, pPinnedBounds]{
            std::copy(reorderedMatches.begin(), reorderedMatches.end(), pPinnedMatches);
            std::copy(bounds.begin(), bounds.end(), pPinnedBounds);
        });
        cudaCheck(cudaMemcpyAsync(pMatches, pPinnedMatches, sizeof(rmf::PtPair) * matches.size(), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(pBounds, pPinnedBounds, sizeof(uint32_t) * (nbCells + 1), cudaMemcpyHostToDevice, stream));
        cudaRansacMatchFilter(pVotes, pAffine, pMatches, cast32u(matches.size()), pBounds, cols, rows, nbRansacTests, cellWidth * threshold, tryOtherAffines, stream);
        cudaCheck(cudaMemcpyAsync(pPinnedVotes, pVotes, sizeof(pVotes[0]) * matches.size(), cudaMemcpyDeviceToHost, stream));
        launchCudaHostFunc(stream, [&votes, pPinnedVotes]{
            std::copy_n(pPinnedVotes, votes.size(), votes.begin());
        });
    }
#if 1
    cudapp::fiberSyncCudaStream(stream);
#else
    cudaCheck(cudaStreamSynchronize(stream));
#endif
    std::vector<bool> results(votes.size());
    for (uint32_t i = 0; i < votes.size(); i++) {
        results.at(flatCellIndices.at(i)) = (votes.at(i) >= minVotes);
    }
    return results;
}
} // namespace rsfm
