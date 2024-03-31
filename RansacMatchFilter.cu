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

#include <cuda_hint.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <kmat.h>
#include <cpp_utils.h>
#include <curand_kernel.h>
#include <cuda/barrier>
#include <cudaExtIntrinsics.cuh>
#include <cooperative_groups.h>
#include <macros.h>
#include <cuda_utils.h>
#include "RansacMatchFilter.cuh"
namespace cg = cooperative_groups;

namespace rsfm::rmf
{

static constexpr uint32_t ctaSize = 128;
// @fixme: PG-2 this threshold-based filter is critical. This may indicate that our image pair solver / IncreModel.
// findings: It causes bad perf for pair solver due to large number of ransac tests. The noise also causes difficulty when merging models.
//
// are not robust enough against outliers
static constexpr uint32_t minNbInliersPerCell = 6;

namespace {

// see glibc linear congruential generator 
__device__ __forceinline__
int32_t minstd_rand (int32_t& state)
{
    state = ((state * 1103515245) + 12345) & 0x7fffffff;
    return state;
}

struct CtaCells
{
    __device__ __forceinline__
    void init(const uint32_t* bounds, uint32_t cols, uint32_t rows) {
        if (threadIdx.x < 9) {
            const uint32_t i = threadIdx.x / 3;
            const uint32_t j = threadIdx.x - i * 3;
            const int32_t m = int(blockIdx.y) - 1 + i;
            const int32_t n = int(blockIdx.x) - 1 + j;
            if (inRange(m, 0, int(rows)) && inRange(n, 0, int(cols))) {
                uint32_t idx = m * cols + n;
                const auto beg = bounds[idx];
                const auto end = bounds[idx + 1];
                cellStart[threadIdx.x] = beg;
                cellSize[threadIdx.x] = end - beg;
            }
            else {
                cellStart[threadIdx.x] = 0;
                cellSize[threadIdx.x] = 0;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t acc = 0;
            for (int i = 0; i < 9; i++) {
                acc += cellSize[i];
            }
            cellSizeSum = acc;
        }
        __syncthreads();
    }
    __device__ __forceinline__
    bool checkInlier(const kmat<float, 2, 3>& trans, const PtPair& p, float sqrThreshold) const {
        kmat<float, 2> diff = trans.block<2, 2>(0, 0) * p.left + trans.col(2) - p.right;
        return diff.sqrNorm() < sqrThreshold;
    }
    template <typename Func>
    __device__ __forceinline__
    void checkInliers(const kmat<float, 2, 3>& trans, const PtPair* __restrict__ matches, float threshold /* = cellWidth*0.15f */, Func&& func) const {
        const float sqrScale = trans(0, 0) * trans(1, 1) - trans(0, 1) * trans(1, 0);
        const float scale = std::sqrt(sqrScale);
        const float sqrThreshold = square(threshold * std::min(scale, 1.f));
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            const uint32_t idxBeg = cellStart[i];
            const uint32_t idxEnd = idxBeg + cellSize[i];
            for (uint32_t idx = idxBeg; idx < idxEnd; idx++) {
                const auto p = matches[idx];
                const bool isInlier = checkInlier(trans, p, sqrThreshold);
                func(idx, isInlier);
            }
        }
    }
    __device__ __forceinline__
    uint32_t getNbInliers(const kmat<float, 2, 3>& trans, const PtPair* __restrict__ matches, float threshold /* = cellWidth*0.15f */) const {
        uint32_t counter = 0;
        checkInliers(trans, matches, threshold, [&counter](uint32_t idxMatch, bool isInlier) mutable {
            unused(idxMatch);
            if (isInlier) {
                counter++;
            }
        });
        return counter;
    }

    __device__ __forceinline__
    uint32_t sampleIdxToMatchIdx(uint32_t idx) const {
        // find which cell idx should fall in
        uint32_t acc = 0;
        uint32_t idxMatch = ~0U;
        #pragma unroll
        for (uint32_t i = 0; i < 9; i++) {
            if (cellSize[i] == 0) {
                continue;
            }
            uint32_t accNext = acc + cellSize[i];
            if (idx < accNext) {
                idxMatch = cellStart[i] + (idx - acc);
                break;
            }
            acc = accNext;
        }
        return idxMatch;
    }


    uint32_t cellStart[9]; // 3x3
    uint32_t cellSize[9]; // 3x3
    uint32_t cellSizeSum;
};

struct CtaReducer
{
    uint32_t mNbInliers[ctaSize / 2];
    kmat<float, 2, 2> mAffineLinear[ctaSize / 2];
    kmat<float, 2> mAffineShift[ctaSize / 2];

    __device__ __forceinline__
    void store(uint32_t idx, uint32_t nbInliers, kmat<float, 2, 3> affine) {
        mNbInliers[idx] = nbInliers;
        mAffineLinear[idx] = affine.block<2, 2>(0, 0);
        mAffineShift[idx] = affine.col(2);
    }
    __device__ __forceinline__
    kmat<float, 2, 3> loadAffine(uint32_t idx) {
        kmat<float, 2, 3> dst;
        dst.assignBlock(0, 0, mAffineLinear[idx]);
        dst.assignBlock(0, 2, mAffineShift[idx]);
        return dst;
    }

    // result is available in threadIdx.x == 0.
    __device__ __forceinline__
    void findMax(uint32_t& nbInliers, kmat<float, 2, 3>& affine) {
        const auto tid = threadIdx.x;
        static_assert((ctaSize & (ctaSize - 1)) == 0);
        //@fixme: use warp reduction when mid is 16.
        #pragma unroll
        for (uint32_t mid = ctaSize / 2; mid != 0; mid /= 2) {
            if (tid >= mid && tid < mid * 2) {
                store(tid - mid, nbInliers, affine);
            }
            __syncthreads();
            if (tid < mid) {
                if (nbInliers < mNbInliers[tid]) {
                    nbInliers = mNbInliers[tid];
                    affine = loadAffine(tid);
                }
            }
        }
    }
    __device__ __forceinline__
    void broadcastMax(uint32_t& nbInliers, kmat<float, 2, 3>& affine)
    {
        if (threadIdx.x == 0) {
            store(0, nbInliers, affine);
        }
        __syncthreads();
        nbInliers = mNbInliers[0];
        affine = loadAffine(0);
    }
};
}

__global__ void kernelFindBestAffineForCells(
    kmat<float, 2, 3>* __restrict__ affine2, // length is cols * rows
    const PtPair* __restrict__ matches, uint32_t nbMatches,
    const uint32_t* __restrict__ bounds, // length is cols*rows+1
    uint32_t cols, uint32_t rows, uint32_t nbRansacTests,
    float threshold // cellWidth * 0.15f
    )
{
    __shared__ CtaCells cells;
    cells.init(bounds, cols, rows);

    if (cells.cellSizeSum < 9) {
        return;
    }

    kmat<float, 2, 3> thrdBestTrans({1, 0, 0, 0, 1, 0});
    uint32_t thrdBestNbInliers = 0u;

    const uint32_t blockRank = blockIdx.y * cols + blockIdx.x;
    const auto gridTid = ctaSize * blockRank + threadIdx.x;
    int32_t rngState = clock() + gridTid * 22695477u;
    __shared__ union IndicesAndReducerUnion {
        uint32_t indices[3][ctaSize]; // 3 elements per-thread. alternative is to use alloca()
        CtaReducer reducer;
    } indicesAndReducer;
    auto& indices = indicesAndReducer.indices;
    auto& reducer = indicesAndReducer.reducer;
    for (uint32_t n = threadIdx.x; n < nbRansacTests; n += ctaSize) {
        // pick 3 random non-duplicate matches
        {
            uint32_t m = 0;
            while (m < 3) {
                const uint32_t idx = uint32_t(minstd_rand(rngState)) % cells.cellSizeSum;
                bool isDuplicate = false;
                for (uint32_t k = 0; k < m; k++) {
                    if (indices[k][threadIdx.x] == idx) {
                        isDuplicate = true;
                        break;
                    }
                }
                if (!isDuplicate) {
                    assert(idx < cells.cellSizeSum);
                    indices[m][threadIdx.x] = idx;
                    assert(indices[m][threadIdx.x] < cells.cellSizeSum);
                    m++;
                }
            }
        }
        
        kmat<float, 3, 3> A;
        kmat<float, 3, 2> b;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            assert(indices[i][threadIdx.x] < cells.cellSizeSum);
            const uint32_t idxMatch = cells.sampleIdxToMatchIdx(indices[i][threadIdx.x]);
            assert(idxMatch < nbMatches);
            auto p = matches[idxMatch];
            A.assignBlock(i, 0, p.left.transpose());
            A(i, 2) = 1;
            b.assignBlock(i, 0, p.right.transpose());
        }
        const kmat<float, 2, 3> trans = solveGaussElim(A, b).transpose();
        const float sqrScale = trans(0, 0) * trans(1, 1) - trans(0, 1) * trans(1, 0);
        // we do not allow negative determinant, as that means flipping
        if(sqrScale < square(0.25f) || sqrScale > square(4.f) || !std::isfinite(sqrScale)) {
            continue;
        }
        uint32_t nbInliers = cells.getNbInliers(trans, matches, threshold);
        if (nbInliers > thrdBestNbInliers) {
            thrdBestNbInliers = nbInliers;
            thrdBestTrans = trans;
        }
    }

    // cta reduction
    uint32_t ctaBestNbInliers = thrdBestNbInliers;
    kmat<float, 2, 3> ctaBestTrans = thrdBestTrans;
    __syncthreads(); // make sure all usage of indicesAndReducer.indices has finished;
    reducer.findMax(ctaBestNbInliers, ctaBestTrans);
    // store best affine2
    if (threadIdx.x == 0) {
        affine2[blockRank] = ctaBestTrans;
    }
}

__global__ void kernelTryBestAffineOfOtherCellsAndVote(
    kmat<float, 2, 3>* __restrict__ dstAffine2, // length is cols * rows. Optional. nullptr if you don't want this.
    uint8_t* __restrict__ votes, // length is same as matches
    const kmat<float, 2, 3>* __restrict__ srcAffine2, // length is cols * rows
    const PtPair* __restrict__ matches, uint32_t nbMatches,
    const uint32_t* __restrict__ bounds, // length is cols*rows+1
    uint32_t cols, uint32_t rows,
    float threshold // cellWidth * 0.15f
    )
{
    __shared__ CtaCells cells;
    cells.init(bounds, cols, rows);
    const uint32_t nbCells = cols * rows;
    const uint32_t blockRank = blockIdx.y * cols + blockIdx.x;
    
    // test solution of all cells for the current cell
    kmat<float, 2, 3> thrdBestTrans = srcAffine2[blockRank];
    uint32_t thrdBestNbInliers = 0u;
    auto getNbInliers = [&](const kmat<float, 2, 3>& trans){
        return cells.getNbInliers(trans, matches, threshold);
    };
    for (uint32_t n = threadIdx.x; n < nbCells; n += ctaSize) {
        const kmat<float, 2, 3> trans = srcAffine2[n];
        uint32_t nbInliers = getNbInliers(trans);
        if (nbInliers > thrdBestNbInliers) {
            thrdBestNbInliers = nbInliers;
            thrdBestTrans = trans;
        }
    }

    // cta reduction
    __shared__ CtaReducer reducer;
    uint32_t ctaBestNbInliers = thrdBestNbInliers;
    kmat<float, 2, 3> ctaBestTrans = thrdBestTrans;
    reducer.findMax(ctaBestNbInliers, ctaBestTrans);

    // store cta best affine2
    if (threadIdx.x == 0 && dstAffine2 != nullptr) {
        dstAffine2[blockRank] = ctaBestTrans;
    }

    // compute votes
    if (votes != nullptr && ctaBestNbInliers >= minNbInliersPerCell)
    {
        reducer.broadcastMax(ctaBestNbInliers, ctaBestTrans);
        const auto cellSizeSum = cells.cellSizeSum;
        const float sqrScale = ctaBestTrans(0, 0) * ctaBestTrans(1, 1) - ctaBestTrans(0, 1) * ctaBestTrans(1, 0);
        const float scale = std::sqrt(sqrScale);
        const float sqrThreshold = square(threshold * std::min(scale, 1.f));
        for (uint32_t idx = threadIdx.x; idx < cellSizeSum; idx += ctaSize) {
            const uint32_t idxMatch = cells.sampleIdxToMatchIdx(idx);
            const bool isInlier = cells.checkInlier(ctaBestTrans, matches[idxMatch], sqrThreshold);
            if (isInlier) {
                cudapp::atomicIncOne(&votes[idxMatch]);
            }
        }
    }
}


__global__ void kernelComputeVotes(
    uint8_t* __restrict__ votes, // length is same as matches
    const kmat<float, 2, 3>* __restrict__ affine2, // length is cols * rows
    const PtPair* __restrict__ matches, uint32_t nbMatches,
    const uint32_t* __restrict__ bounds, // length is cols*rows+1
    uint32_t cols, uint32_t rows,
    float threshold // cellWidth * 0.15f
    )
{
    __shared__ uint32_t nbInliers;
    if (threadIdx.x == 0) {
        nbInliers = 0;
    }
    __syncthreads();
    __shared__ CtaCells cells;
    cells.init(bounds, cols, rows);
    const uint32_t blockRank = blockIdx.y * cols + blockIdx.x;
    
    kmat<float, 2, 3> trans = affine2[blockRank];


    // compute votes
    {
        const auto cellSizeSum = cells.cellSizeSum;
        const float sqrScale = trans(0, 0) * trans(1, 1) - trans(0, 1) * trans(1, 0);
        const float scale = std::sqrt(sqrScale);
        const float sqrThreshold = square(threshold * std::min(scale, 1.f));
        //@fixme: store nbInliers in the previous kernel and load it here, instead of re-compute it here and revert
        for (uint32_t idx = threadIdx.x; idx < cellSizeSum; idx += ctaSize) {
            const uint32_t idxMatch = cells.sampleIdxToMatchIdx(idx);
            assert(idxMatch < nbMatches); unused(nbMatches);
            const bool isInlier = cells.checkInlier(trans, matches[idxMatch], sqrThreshold);
            if (isInlier) {
                cudapp::atomicIncOne(&votes[idxMatch]);
                atomicAdd(&nbInliers, 1);
            }
        }
        __syncthreads();
        if (nbInliers < minNbInliersPerCell) {
            // revert
            for (uint32_t idx = threadIdx.x; idx < cellSizeSum; idx += ctaSize) {
                const uint32_t idxMatch = cells.sampleIdxToMatchIdx(idx);
                assert(idxMatch < nbMatches); unused(nbMatches);
                const bool isInlier = cells.checkInlier(trans, matches[idxMatch], sqrThreshold);
                if (isInlier) {
                    cudapp::atomicDecOne(&votes[idxMatch]);
                }
            }
        }
    }
}

// votes must be padded to multiples of 4.
void cudaRansacMatchFilter(uint8_t* votes, // length is same as matches
    kmat<float, 2, 3>* affineScratch, // length is cols * rows
    const PtPair* matches,
    uint32_t nbMatches,
    const uint32_t* bounds, // length is cols*rows+1
    uint32_t cols, uint32_t rows, uint32_t nbRansacTests,
    float threshold, bool tryOtherSolutions, cudaStream_t stream)
{
    ASSERT(reinterpret_cast<std::uintptr_t>(votes) % 4 == 0);
    cudaCheck(cudaMemsetAsync(votes, 0, sizeof(votes[0]) * nbMatches, stream));

    if (nbMatches < 16) {
        return;
    }

    const dim3 gridShape{cols, rows};
    launchKernel(&kernelFindBestAffineForCells, gridShape, ctaSize, size_t{0}, stream, affineScratch, matches, nbMatches, bounds, cols, rows, nbRansacTests, threshold);
    
    if (tryOtherSolutions) {
        launchKernel(&kernelTryBestAffineOfOtherCellsAndVote, gridShape, ctaSize, size_t{0}, stream, static_cast<kmat<float, 2, 3>*>(nullptr), votes, static_cast<const kmat<float, 2, 3>*>(affineScratch), matches, nbMatches, bounds, cols, rows, threshold);
    }
    else {
        launchKernel(&kernelComputeVotes, gridShape, ctaSize, size_t{0}, stream, votes, static_cast<const kmat<float, 2, 3>*>(affineScratch), matches, nbMatches, bounds, cols, rows, threshold);
    }
    
}

} // namespace rsfm::rmf
