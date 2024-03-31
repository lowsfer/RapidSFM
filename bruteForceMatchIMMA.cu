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
#include "bruteForceMatch.cuh"
#include <type_traits>
#include <ldg.cuh>
#include <ptr.h>
#include "immaUtils.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define TRAITS_IMPORT(x) static constexpr decltype(Traits::x) x = Traits::x

template <typename T, typename OrigType = T>
using Ptr = cudapp::Ptr<T, OrigType>;

#define USE_DOUBLE_BUFFER 1
#define PREFETCH_DESC_SQR_NORM 1

namespace rsfm
{
namespace imma
{

// constexpr DescElemType descElemType_ = DescElemType::kU4;
// constexpr size_t descDims_ = 128;
// constexpr DistanceStyle distStyle_ = DistanceStyle::kL2;
// constexpr bool bidirectional_ = true;
template<DescElemType descElemType_, size_t descDims_, DistanceStyle distStyle_, bool bidirectional_, int mmaM_, int mmaN_, int mmaK_>
struct MatchTraits : DescTraits<descElemType_, descDims_, distStyle_>{
    using Base = DescTraits<descElemType_, descDims_, distStyle_>;
    static constexpr bool bidirectional = bidirectional_;
    static constexpr DistanceStyle distStyle = Base::distStyle;
    static constexpr DescElemType descElemType = Base::descElemType;
    static constexpr int descDims = Base::descDims;
    using Distance = typename Base::Distance;
    using Word = typename Base::Word;
    static constexpr bool isDistNonNegative = true;
    static constexpr int descWords = Base::descWords;
    using Descriptor = typename Base::Descriptor;
    using Task = BruteForceMatchTask<descElemType, descDims, distStyle>;
    static constexpr HW ctaWarps = {2, 4};
    static constexpr int warp_size = imma::warp_size; // warpSize is already defined and it's not constexpr
    static constexpr int nbBanks = imma::nbBanks;
    static constexpr HW warpShape = {8, 4};
#if !(defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 3 && __CUDACC_VER_MINOR__ <= 8))
    // cuda-11.3/11.4/11.5/11.6 do not consider warpShape.w/.w constexpr
    static_assert(warp_size == warpShape.w * warpShape.h);
#endif
    static constexpr int ctaSize = warp_size * ctaWarps.w * ctaWarps.h;
    static constexpr int mmaM = mmaM_;
    static constexpr int mmaN = mmaN_;
    static constexpr int mmaK = mmaK_;

    using MMAType = MMA<descElemType, mmaM, mmaN, mmaK>;

    // MNK config for 1 register for A, 1 register for B and two register for accumulators
    static constexpr int baseMmaM = MMAType::baseMmaM;
    static constexpr int baseMmaN = MMAType::baseMmaN;
    static constexpr int baseMmaK = MMAType::baseMmaK;

    static constexpr HW warpTileAtom = {mmaM, mmaN};
    static constexpr HW warpTileUnroll = {4, 4};
    static constexpr HW warpTile = warpTileAtom * warpTileUnroll;

    static constexpr HW ctaTile = warpTile * ctaWarps;

    using BestMatchCta = ReducedBestMatchType<false, 8, 24, true>;
    using Index = typename BestMatchCta::Index;
    using BestMatchGlobal = BestMatchType<Distance, true, true>;

    using AccAtom = typename MMAType::TileAcc;
    using AtomA = typename MMAType::TileA;
    using AtomB = typename MMAType::TileB;

    static constexpr HW accAtomShape = {AccAtom::dimension, AccAtom::Elem::dimension};

    __device__ __forceinline__
    static AccAtom computeAtom(const AccAtom& init, const AtomA& a, const AtomB& b) {
        return MMAType::run(a, b, init);
    }
    static constexpr int nbRegBufA = USE_DOUBLE_BUFFER ? 2 : 1;
    static constexpr int nbRegBufB = USE_DOUBLE_BUFFER ? 2 : 1;

    __device__ __forceinline__
    static int getIdxWarpInCta() {
        return threadIdx.x / warp_size;
    }
    __device__ __forceinline__
    static HW getWarpLocInCta() {
        const auto idxWarpInCta = getIdxWarpInCta();
        return {idxWarpInCta / ctaWarps.w, idxWarpInCta % ctaWarps.w};
    }
    __device__ __forceinline__
    static HW getThrdLocInWarp() {
        return {int(lane_id()) / warpShape.w, int(lane_id()) % warpShape.w};
    }
    __device__ __forceinline__
    static HW getCtaLocInGrid() {
        return {int(blockIdx.y), int (blockIdx.x)};
    }
};


struct TraitsIMMASift4b : MatchTraits<DescElemType::kU4, 128, DistanceStyle::kL2, true, 16, 8, 64>{};
struct TraitsIMMASift8b : MatchTraits<DescElemType::kU8, 128, DistanceStyle::kL2, true, 16, 8, 32>{};

template <typename Traits>
struct WarpTask : public Traits
{
    using SMemInTileA = SMemInTile<Traits::descElemType, Traits::descDims, Traits::distStyle, Traits::ctaTile.h>;
    using SMemInTileB = SMemInTile<Traits::descElemType, Traits::descDims, Traits::distStyle, Traits::ctaTile.w>;

    static constexpr int nbRegBufA = Traits::nbRegBufA;
    static constexpr int nbRegBufB = Traits::nbRegBufB;
    static constexpr HW warpTileUnroll = Traits::warpTileUnroll;
    static constexpr HW accAtomShape = Traits::accAtomShape;
    TRAITS_IMPORT(mmaM);
    TRAITS_IMPORT(mmaN);
    TRAITS_IMPORT(mmaK);
    TRAITS_IMPORT(warpTile);
    TRAITS_IMPORT(warpTileAtom);
    TRAITS_IMPORT(warpShape);
    using BestMatchCta = typename Traits::BestMatchCta;
    using MMAType = typename Traits::MMAType;
    using Distance = typename Traits::Distance;
    using Index = typename Traits::Index;


    __device__ __forceinline__
    WarpTask(SMemInTileA& smemA_, SMemInTileB& smemB_, bool isBidirectional_)
        : smemA{smemA_}
        , smemB{smemB_}
        , isBidirectional{isBidirectional_}
    {
        #pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
            #pragma unroll
            for(int w = 0; w < warpTileUnroll.w; w++) {
                #pragma unroll
                for (int i = 0; i < accAtomShape.h; i++) {
                    #pragma unroll
                    for (int j = 0; j < accAtomShape.w; j++) {
                        acc[h][w][i][j] = 0u;
                    }
                }
            }
        }
    }

    const SMemInTileA& smemA;
    const SMemInTileB& smemB;
    bool isBidirectional;
    KArray<typename Traits::AtomA, nbRegBufA> a;
    KArray<typename Traits::AtomB, nbRegBufB, Traits::warpTileUnroll.w> b;
    KArray<typename Traits::AccAtom, Traits::warpTileUnroll.h, Traits::warpTileUnroll.w> acc;

    __device__ __forceinline__
    HW getCoordInWarp(const HW& idxUnroll, const HW& idxInAtom) const {
        constexpr HW warpTileAtom = Traits::warpTileAtom;
        const HW thrdLocInWarp = Traits::getThrdLocInWarp();
        return warpTileAtom * idxUnroll + HW{
            MMAType::baseDstNbAcc.h * warpShape.h * idxInAtom.h + thrdLocInWarp.h,
            MMAType::baseDstNbAcc.w * thrdLocInWarp.w + idxInAtom.w
        };
    }

    __device__ __forceinline__
    HW getCoordInCta(const HW& idxUnroll, const HW& idxInAtom) const {
        constexpr HW warpTile = Traits::warpTile;
        return warpTile * Traits::getWarpLocInCta() + getCoordInWarp(idxUnroll, idxInAtom);
    }

    __device__ __forceinline__
    void computeRow(int idxBufA, int idxBufB, int idxRow) {
#pragma unroll
        for (int i = 0; i < warpTileUnroll.w; i++) {
            acc[idxRow][i] = computeAtom(acc[idxRow][i], a[idxBufA], b[idxBufB][i]);
        }
    }

    __device__ __forceinline__
    void smemLdA(int idxBufA, int idxRow, int idxKBlk) {
        a[idxBufA] = smemA.warpLoadMatrix<mmaM, mmaK / MMAType::elemsPerWord>(warpTileUnroll.h * Traits::getWarpLocInCta().h + idxRow, idxKBlk);
    }
    __device__ __forceinline__
    void smemLdB(int idxBufB, int idxKBlk) {
        #pragma unroll
        for (int i = 0; i < warpTileUnroll.w; i++) {
            b[idxBufB][i] = smemB.warpLoadMatrix<mmaN, mmaK / MMAType::elemsPerWord>(warpTileUnroll.w * Traits::getWarpLocInCta().w + i, idxKBlk);
        }
    }

    __device__ __forceinline__
    void finishAcc(const Ptr<const uint32_t>& ctaSqrNormDescA, const Ptr<const uint32_t>& ctaSqrNormDescB)
    {
        const HW warpLocInCta = Traits::getWarpLocInCta();
        const auto warpSqrNormDescA = ctaSqrNormDescA + warpTile.h * warpLocInCta.h;
        const auto warpSqrNormDescB = ctaSqrNormDescB + warpTile.w * warpLocInCta.w;

        KArray<uint32_t, Traits::warpTileUnroll.h, Traits::accAtomShape.h> sqrNormA;
        // These bad values help us disgard out-of-bound padding accumulators. (We handle 128x128 tile per CTA.)
        constexpr uint32_t badSqrNormA = BestMatchCta::worstValue / 2u;
        constexpr uint32_t badSqrNormB = BestMatchCta::worstValue - badSqrNormA;
        #pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
            const auto atomSqrNormDescA = warpSqrNormDescA + warpTileAtom.h * h;
            #pragma unroll
            for (int i = 0; i < accAtomShape.h; i++) {
                const auto ptr = atomSqrNormDescA + warpShape.h * i + Traits::getThrdLocInWarp().h;
                sqrNormA[h][i] = ptr.isInBound() ? cudapp::ldg(ptr) : badSqrNormA;
            }
        }

        KArray<uint32_t, Traits::warpTileUnroll.w, Traits::accAtomShape.w> sqrNormB;
        #pragma unroll
        for(int w = 0; w < warpTileUnroll.w; w++) {
            const auto atomSqrNormDescB = warpSqrNormDescB + warpTileAtom.w * w;
            for (int j = 0; j < accAtomShape.w; j++) {
                const auto ptr = atomSqrNormDescB + accAtomShape.w * Traits::getThrdLocInWarp().w + j;
                sqrNormB[w][j] = ptr.isInBound() ? cudapp::ldg(ptr) : badSqrNormB;
            }
        }

        #pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
            #pragma unroll
            for(int w = 0; w < warpTileUnroll.w; w++) {
                #pragma unroll
                for (int i = 0; i < accAtomShape.h; i++) {
                    #pragma unroll
                    for (int j = 0; j < accAtomShape.w; j++) {
                        acc[h][w][i][j] = sqrNormA[h][i] + sqrNormB[w][j] - acc[h][w][i][j] * 2;

                        const auto coordInCta = getCoordInCta(HW{h,w}, HW{i,j});
                        unused(coordInCta);
                        assert(sqrNormA[h][i] == ((ctaSqrNormDescA + coordInCta.h).isInBound() ? ctaSqrNormDescA[coordInCta.h] : badSqrNormA));
                        assert(sqrNormB[w][j] == ((ctaSqrNormDescB + coordInCta.w).isInBound() ? ctaSqrNormDescB[coordInCta.w] : badSqrNormB));
                    }
                }
            }
        }
    }

    __device__ __forceinline__
    std::pair<KArray<BestMatchCta, Traits::warpTileUnroll.h, Traits::accAtomShape.h>,
        KArray<BestMatchCta, Traits::warpTileUnroll.w, Traits::accAtomShape.w>>
    computeWarpBestMatches() {
        KArray<BestMatchCta, Traits::warpTileUnroll.h, Traits::accAtomShape.h> bestMatchA;
        KArray<BestMatchCta, Traits::warpTileUnroll.w, Traits::accAtomShape.w> bestMatchB;
        // find best matches in thread data
#pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
#pragma unroll
            for (int w = 0; w < warpTileUnroll.w; w++) {
#pragma unroll
                for (int i = 0; i < accAtomShape.h; i++) {
#pragma unroll
                    for (int j = 0; j < accAtomShape.w; j++) {
                        const Distance distance = acc[h][w][i][j];
                        const HW coordInCta = getCoordInCta(HW{h,w}, HW{i,j});
                        {
                            const Index idxBInCta = coordInCta.w;
                            const BestMatchCta m = BestMatchCta::makeInstance(distance, idxBInCta);
                            if (w == 0 && j == 0) {
                                bestMatchA[h][i] = m;
                            }
                            else {
                                bestMatchA[h][i].updateRegister(m);
                            }
                        }
                        if (isBidirectional)
                        {
                            const Index idxAInCta = coordInCta.h;
                            const BestMatchCta m = BestMatchCta::makeInstance(distance, idxAInCta);
                            if (h == 0 && i == 0) {
                                bestMatchB[w][j] = m;
                            }
                            else {
                                bestMatchB[w][j].updateRegister(m);
                            }
                        }
                    }
                }
            }
        }

        // find best matches in warp
        {
    #pragma unroll
            for (int h = 0; h < warpTileUnroll.h; h++) {
    #pragma unroll
                for (int i = 0; i < accAtomShape.h; i++) {
                    for (unsigned x = 1; x < warpShape.w; x <<= 1) {
                        auto& thisMatch = bestMatchA[h][i];
                        BestMatchCta other;
                        other.raw = __shfl_xor_sync(~0u, thisMatch.raw, x);
                        thisMatch.updateRegister(other);
                    }
                }
            }
        }
        if (isBidirectional) {
    #pragma unroll
            for (int w = 0; w < warpTileUnroll.w; w++) {
    #pragma unroll
                for (int j = 0; j < accAtomShape.w; j++) {
                    for (unsigned x = warpShape.w; x < warp_size; x <<= 1) {
                        auto& thisMatch = bestMatchB[w][j];
                        BestMatchCta other;
                        other.raw = __shfl_xor_sync(~0u, thisMatch.raw, x);
                        thisMatch.updateRegister(other);
                    }
                }
            }
        }
        return std::make_pair(bestMatchA, bestMatchB);
    }
};

template <typename Traits>
struct CtaTask : Traits
{
    using SMemInTileA = typename WarpTask<Traits>::SMemInTileA;
    using SMemInTileB = typename WarpTask<Traits>::SMemInTileB;
    using Descriptor = typename Traits::Descriptor;
    using Task = typename Traits::Task;
    using BestMatchCta = typename Traits::BestMatchCta;
    using BestMatchGlobal = typename Traits::BestMatchGlobal;
    static constexpr int ctaSize = Traits::ctaSize;
    static constexpr int nbRegBufA = Traits::nbRegBufA;
    static constexpr int nbRegBufB = Traits::nbRegBufB;
    TRAITS_IMPORT(descDims);
    TRAITS_IMPORT(mmaK);
    TRAITS_IMPORT(mmaM);
    TRAITS_IMPORT(mmaN);
    TRAITS_IMPORT(warpTileUnroll);
    TRAITS_IMPORT(accAtomShape);

    __device__ __forceinline__
    CtaTask(const Task& task, SMemInTileA& a, SMemInTileB& b,
        KArray<typename Traits::BestMatchCta, Traits::ctaTile.h>& smemBestMatchA,
        KArray<typename Traits::BestMatchCta, Traits::ctaTile.w>& smemBestMatchB)
    : task{task}
    , smemA{a}
    , smemB{b}
    , warpTask(a, b, isBidirectional)
    , smemBestMatchA(smemBestMatchA)
    , smemBestMatchB(smemBestMatchB)
    {
		assert(!isBidirectional || task.bestMatchB != nullptr);
    }
    const Task& task;
    static constexpr bool isBidirectional = Traits::bidirectional;
    SMemInTileA& smemA;
    SMemInTileB& smemB;
    WarpTask<Traits> warpTask;
    KArray<typename Traits::BestMatchCta, Traits::ctaTile.h>& smemBestMatchA;
    KArray<typename Traits::BestMatchCta, Traits::ctaTile.w>& smemBestMatchB;

    __device__ __forceinline__
    void prologue() {
        constexpr HW ctaTile = Traits::ctaTile;
        const HW idxDescBase = ctaTile * Traits::getCtaLocInGrid();
        const HW idxDescEndIfFull = idxDescBase + ctaTile;
        const auto ctaDescA = cudapp::Ptr<const Descriptor>(task.descA, task.nbDescA, idxDescBase.h);
        if (idxDescEndIfFull.h <= task.nbDescA) {
            smemA.ctaFill<ctaSize, true>(ctaDescA);
        }
        else {
            smemA.ctaFill<ctaSize, false>(ctaDescA);
        }
        const cudapp::Ptr<const Descriptor> ctaDescB {task.descB, task.nbDescB, idxDescBase.w};
        if (idxDescEndIfFull.w <= task.nbDescB) {
            smemB.ctaFill<ctaSize, true>(ctaDescB);
        }
        else {
            smemB.ctaFill<ctaSize, false>(ctaDescB);
        }
        __syncthreads();
#if PREFETCH_DESC_SQR_NORM
        prefetchDescSqrNorm();
#endif
    }
#if USE_DOUBLE_BUFFER
    __device__ __forceinline__
    void mainloop() {
        static_assert(nbRegBufA != 1 && nbRegBufB != 1);
        warpTask.smemLdB(0, 0);
        warpTask.smemLdA(0, 0, 0);
#pragma unroll
        for (int k = 0; k < descDims / mmaK; k++) {
#pragma unroll
            for (int i = 0; i < warpTileUnroll.h; i++) {
                // prefetch with double buffer
                if (i + 1 < warpTileUnroll.h) {
                    warpTask.smemLdA((i + 1) % nbRegBufA, i + 1, k);
                }
                else if (k + 1 < descDims / mmaK) {
                    warpTask.smemLdA((i + 1) % nbRegBufA, (i + 1) % warpTileUnroll.h, k + 1);
                    warpTask.smemLdB((k + 1) % nbRegBufB, k + 1);
                }
                // compute
                warpTask.computeRow(i % nbRegBufA, k % nbRegBufB, i);
            }
        }
    }
#else
    __device__ __forceinline__
    void mainloop() {
        static_assert(nbRegBufA == 1 && nbRegBufB == 1);
#pragma unroll(1)
        for (int k = 0; k < descDims / mmaK; k++) {
            warpTask.smemLdB(0, k);
#pragma unroll
            for (int i = 0; i < warpTileUnroll.h; i++) {
                warpTask.smemLdA(0, i, k);
                warpTask.computeRow(0, 0, i);
            }
        }
    }
#endif
    __device__ __forceinline__
    void epilogue(){
        constexpr HW ctaTile = Traits::ctaTile;
        const HW idxDescBase = ctaTile * Traits::getCtaLocInGrid();
        const Ptr<const uint32_t> ctaSqrNormDescA {task.sqrNormDescA, task.nbDescA, idxDescBase.h};
        const Ptr<const uint32_t> ctaSqrNormDescB {task.sqrNormDescB, task.nbDescB, idxDescBase.w};
        warpTask.finishAcc(ctaSqrNormDescA, ctaSqrNormDescB);

        const auto warpBestMatches = warpTask.computeWarpBestMatches();
        const auto& warpBestMatchA = warpBestMatches.first;
        const auto& warpBestMatchB = warpBestMatches.second;
        const auto thrdLocInWarp = Traits::getThrdLocInWarp();

        __syncthreads(); // required because smemBestMatchA/B overlaps with inTileB
#pragma unroll
        for (int i = 0; i < divUp(ctaTile.h, ctaSize); i++) {
            const int idxDescCta = ctaSize * i + threadIdx.x;
            if (idxDescCta >= ctaTile.h) {
                break;
            }
            smemBestMatchA[idxDescCta].init();
        }
        if (isBidirectional) {
#pragma unroll
            for (int i = 0; i < divUp(ctaTile.w, ctaSize); i++) {
                const int idxDescCta = ctaSize * i + threadIdx.x;
                if (idxDescCta >= ctaTile.w) {
                    break;
                }
                smemBestMatchB[idxDescCta].init();
            }
        }
        __syncthreads();
        if (thrdLocInWarp.w == 0) {
#pragma unroll
            for (int h = 0; h < warpTileUnroll.h; h++) {
#pragma unroll
                for (int i = 0; i < accAtomShape.h; i++) {
                    smemBestMatchA[warpTask.getCoordInCta(HW{h, 0}, HW{i, 0}).h].updateRam(warpBestMatchA[h][i]);
                }
            }
        }
        if (isBidirectional && thrdLocInWarp.h == 0) {
#pragma unroll
            for (int w = 0; w < warpTileUnroll.w; w++) {
#pragma unroll
                for (int j = 0; j < accAtomShape.w; j++) {
                    smemBestMatchB[warpTask.getCoordInCta(HW{0, w}, HW{0, j}).w].updateRam(warpBestMatchB[w][j]);
                }
            }
        }
        __syncthreads();

        const HW ctaLocInGrid = Traits::getCtaLocInGrid();
        const int idxDescBaseA = ctaTile.h * ctaLocInGrid.h;
        const int idxDescBaseB = ctaTile.w * ctaLocInGrid.w;

#pragma unroll
        for (int n = 0; n < divUp(ctaTile.h, ctaSize); n++) {
            const int idxDescCta = ctaSize * n + threadIdx.x;
            if (idxDescCta >= ctaTile.h) {
                break;
            }
            const BestMatchCta ctaBestMatchA = smemBestMatchA[idxDescCta];
            if (idxDescBaseA + idxDescCta < task.nbDescA) {
                reinterpret_cast<BestMatchGlobal&>(task.bestMatchA[idxDescBaseA + idxDescCta]).updateRam(ctaBestMatchA.distance(), ctaBestMatchA.index() + idxDescBaseB);
            }
        }
        if (isBidirectional) {
#pragma unroll
            for (int n = 0; n < divUp(ctaTile.w, ctaSize); n++) {
                const int idxDescCta = ctaSize * n + threadIdx.x;
                if (idxDescCta >= ctaTile.w) {
                    break;
                }
                BestMatchCta ctaBestMatchB = smemBestMatchB[idxDescCta];
                if (idxDescBaseB + idxDescCta < task.nbDescB) {
                    reinterpret_cast<BestMatchGlobal&>(task.bestMatchB[idxDescBaseB + idxDescCta]).updateRam(ctaBestMatchB.distance(), ctaBestMatchB.index() + idxDescBaseA);
                }
            }
        }
    }

    __device__ __forceinline__
    void prefetchDescSqrNorm(){
        auto prefetch = [](const uint32_t* p, int nbRemainingDesc, int maxNbDesc){
            if (nbRemainingDesc < maxNbDesc) {
                #pragma unroll
                for (int i = 0; i < divUp(maxNbDesc, ctaSize); i++) {
                    const int idx = ctaSize * i + threadIdx.x;
                    if (idx < nbRemainingDesc) {
                        prefetchL1(p+idx);
                    }
                }
            }
            else {
                #pragma unroll
                for (int i = 0; i < divUp(maxNbDesc, ctaSize); i++) {
                    const int idx = ctaSize * i + threadIdx.x;
                    prefetchL1(p+idx);
                }
            }
        };

        constexpr HW ctaTile = Traits::ctaTile;
        const HW idxDescBase = ctaTile * Traits::getCtaLocInGrid();
        prefetch(task.sqrNormDescA + idxDescBase.h, int(task.nbDescA) - idxDescBase.h, ctaTile.h);
        prefetch(task.sqrNormDescB + idxDescBase.w, int(task.nbDescB) - idxDescBase.w, ctaTile.h);
    }
};

static constexpr size_t maxNbTasks = 16;

template <typename Traits>
__global__ void
#ifdef NDEBUG
__launch_bounds__(Traits::ctaSize, 2)
#endif
kernelBruteForceMatchIMMA(const std::array<typename Traits::Task, maxNbTasks> tasks) {
    using SMemInTileA = typename WarpTask<Traits>::SMemInTileA;
    using SMemInTileB = typename WarpTask<Traits>::SMemInTileB;

    __shared__ struct {
        typename CtaTask<Traits>::SMemInTileA inTileA;
        union {
            typename CtaTask<Traits>::SMemInTileB inTileB;
            struct {
                KArray<typename Traits::BestMatchCta, Traits::ctaTile.h> bestMatchA;
                KArray<typename Traits::BestMatchCta, Traits::ctaTile.w> bestMatchB;
            };
        };
    } smem;


    const auto idxTask = blockIdx.z;
    assert(idxTask < tasks.size());
    const auto& task = tasks[idxTask];
    __syncthreads();

    constexpr auto ctaTile = Traits::ctaTile;
    const HW idxDescBase = ctaTile * Traits::getCtaLocInGrid();
    if (idxDescBase.h >= task.nbDescA || idxDescBase.w >= task.nbDescB) {
        return;
    }
    CtaTask<Traits> ctaTask{task, smem.inTileA, smem.inTileB, smem.bestMatchA, smem.bestMatchB};
    ctaTask.prologue();
    ctaTask.mainloop();
    ctaTask.epilogue();
}

static constexpr uint32_t ctaSizeForInit = 256;

template <typename Traits>
void launchBruteForceMatchIMMAImpl(const typename Traits::Task* tasks, size_t nbTasks, cudaStream_t stream)
{
    assert(nbTasks <= maxNbTasks);
    uint32_t maxNbDescA = 0;
    uint32_t maxNbDescB = 0;
    for (size_t i = 0; i < nbTasks; i++) {
        maxNbDescA = std::max(maxNbDescA, tasks[i].nbDescA);
        maxNbDescB = std::max(maxNbDescB, tasks[i].nbDescB);
    }
    std::array<typename Traits::Task, maxNbTasks> argTasks{};
    std::copy_n(tasks, nbTasks, argTasks.begin());

    const dim3 gridForInit{divUp(std::max(maxNbDescA, maxNbDescB), ctaSizeForInit), 1, static_cast<uint32_t>(nbTasks)};
    launchKernel(&kernelInitBruteForceMatch<Traits, maxNbTasks, ctaSizeForInit>,
        gridForInit, ctaSizeForInit, 0, stream, argTasks);

    const dim3 gridForMatch {
        divUp(maxNbDescB, Traits::ctaTile.w),
        divUp(maxNbDescA, Traits::ctaTile.h),
        static_cast<uint32_t>(nbTasks)
    };
    launchKernel(&kernelBruteForceMatchIMMA<Traits>, gridForMatch, Traits::ctaSize, 0, stream, argTasks);
}

} // namespace imma

void launchBruteForceMatchIMMA(const typename imma::TraitsIMMASift4b::Task* tasks, size_t nbTasks, cudaStream_t stream)
{
    using Traits = imma::TraitsIMMASift4b;
    static_assert(std::is_same<typename Traits::Task, Sift4bBruteForceMatchTask>::value);
    for (size_t i = 0; i < nbTasks; i += imma::maxNbTasks) {
        imma::launchBruteForceMatchIMMAImpl<Traits>(tasks + i, std::min(imma::maxNbTasks, nbTasks - i), stream);
    }
}

void launchBruteForceMatchIMMA(const typename imma::TraitsIMMASift8b::Task* tasks, size_t nbTasks, cudaStream_t stream)
{
    using Traits = imma::TraitsIMMASift8b;
    static_assert(std::is_same<typename Traits::Task, SiftBruteForceMatchTask>::value);
    for (size_t i = 0; i < nbTasks; i += imma::maxNbTasks) {
        imma::launchBruteForceMatchIMMAImpl<Traits>(tasks + i, std::min(imma::maxNbTasks, nbTasks - i), stream);
    }
}

namespace {
static constexpr int ctaSizeFor8bTo4b = 256;
__global__ void kernelSiftDesc8bTo4b(cudapp::KArray<uint32_t, 16>* __restrict__ dst,
    const cudapp::KArray<uint32_t, 32>* __restrict__ src, uint32_t nbDesc)
{
    const int tid = ctaSizeFor8bTo4b * blockIdx.x + threadIdx.x;
    using SrcLdVec = KArray<uint32_t, 4>;
    using DstStVec = KArray<uint32_t, 2>;
    constexpr int nbVecsPerSrcDesc = sizeof(*src) / sizeof(SrcLdVec);
    constexpr int nbVecsPerDstDesc = sizeof(*dst) / sizeof(DstStVec);
    static_assert(nbVecsPerDstDesc == nbVecsPerSrcDesc);
    constexpr int nbVecsPerDesc = nbVecsPerSrcDesc;
    const int idxDesc = tid / nbVecsPerDesc;
    if (idxDesc >= nbDesc) {
        return;
    }
    const int idxVecInDesc = tid % nbVecsPerDesc;
    const SrcLdVec srcVec = reinterpret_cast<const SrcLdVec*>(&src[idxDesc])[idxVecInDesc];
    DstStVec dstVec;
    for (int i = 0; i < 2; i++) {
        const uint32_t a = ((srcVec[2 * i] & 0xF0F0F0F0u) >> 4);
        const uint32_t b = (srcVec[2 * i + 1] & 0xF0F0F0F0u);
        dstVec[i] = (a | b);
    }
    reinterpret_cast<DstStVec*>(&dst[idxDesc])[idxVecInDesc] = dstVec;
}
}

void launchSiftDesc8bTo4b(cudapp::KArray<uint32_t, 16>* dst,
    const cudapp::KArray<uint32_t, 32>* src, uint32_t nbDesc, cudaStream_t stream)
{
    launchKernel(&kernelSiftDesc8bTo4b, divUp(8 * nbDesc, ctaSizeFor8bTo4b), ctaSizeFor8bTo4b, 0, stream, dst, src, nbDesc);
}

namespace{
static constexpr int ctaSizeFor4bSqrNorm = 256;
__global__ void kernelPreCompSift4bSqrNorm(uint32_t* __restrict__ sqrNorm,
    const cudapp::KArray<uint32_t, 16>* __restrict__ src, uint32_t nbDesc)
{
    constexpr int ctaSize = ctaSizeFor4bSqrNorm;
    const int tid = ctaSize * blockIdx.x + threadIdx.x;
    using LdVec = KArray<uint32_t, 4>;
    constexpr int nbVecsPerDesc = sizeof(*src) / sizeof(LdVec);
    const int idxDesc = tid / nbVecsPerDesc;
    if (idxDesc >= nbDesc) {
        return;
    }
    const int idxVecInDesc = tid % nbVecsPerDesc;
    const LdVec vec = reinterpret_cast<const LdVec*>(&src[idxDesc])[idxVecInDesc];
    uint32_t acc = 0u;
    for (const uint32_t v : vec) {
        const uint32_t a = (v & 0x0F0F0F0Fu);
        const uint32_t b = ((v >> 4) & 0x0F0F0F0Fu);
        acc = __dp4a(a, a, acc);
        acc = __dp4a(b, b, acc);
    }

    const auto grp = cg::tiled_partition<nbVecsPerDesc>(cg::this_thread_block());
    #pragma unroll
    for (unsigned x = 1; x < nbVecsPerDesc; x <<= 1) {
        acc += grp.shfl_xor(acc, x);
    }

    if (grp.thread_rank() == 0){
        sqrNorm[idxDesc] = acc;
    }
}

static constexpr int ctaSizeFor8bSqrNorm = 256;
__global__ void kernelPreCompSift8bSqrNorm(uint32_t* __restrict__ sqrNorm,
    const cudapp::KArray<uint32_t, 32>* __restrict__ src, uint32_t nbDesc)
{
    constexpr int ctaSize = ctaSizeFor8bSqrNorm;
    const int tid = ctaSize * blockIdx.x + threadIdx.x;
    using LdVec = KArray<uint32_t, 4>;
    constexpr int nbVecsPerDesc = sizeof(*src) / sizeof(LdVec);
    const int idxDesc = tid / nbVecsPerDesc;
    if (idxDesc >= nbDesc) {
        return;
    }
    const int idxVecInDesc = tid % nbVecsPerDesc;
    const LdVec vec = reinterpret_cast<const LdVec*>(&src[idxDesc])[idxVecInDesc];
    uint32_t acc = 0u;
    for (const uint32_t v : vec) {
        acc = __dp4a(v, v, acc);
    }

    const auto grp = cg::tiled_partition<nbVecsPerDesc>(cg::this_thread_block());
    #pragma unroll
    for (unsigned x = 1; x < nbVecsPerDesc; x <<= 1) {
        acc += grp.shfl_xor(acc, x);
    }

    if (grp.thread_rank() == 0){
        sqrNorm[idxDesc] = acc;
    }
}
} // namespace unnamed

void launchPreCompSift4bSqrNorm(uint32_t* sqrNorm,
    const cudapp::KArray<uint32_t, 16>* src, uint32_t nbDesc, cudaStream_t stream)
{
    using LdVec = KArray<uint32_t, 4>;
    constexpr int nbVecsPerDesc = sizeof(*src) / sizeof(LdVec);
    constexpr int ctaSize = ctaSizeFor4bSqrNorm;
    launchKernel(&kernelPreCompSift4bSqrNorm, divUp(nbVecsPerDesc * nbDesc, ctaSize), ctaSize, 0, stream, sqrNorm, src, nbDesc);
}

void launchPreCompSiftSqrNorm(uint32_t* sqrNorm,
    const cudapp::KArray<uint32_t, 32>* src, uint32_t nbDesc, cudaStream_t stream)
{
    using LdVec = KArray<uint32_t, 4>;
    constexpr int nbVecsPerDesc = sizeof(*src) / sizeof(LdVec);
    constexpr int ctaSize = ctaSizeFor4bSqrNorm;
    launchKernel(&kernelPreCompSift8bSqrNorm, divUp(nbVecsPerDesc * nbDesc, ctaSize), ctaSize, 0, stream, sqrNorm, src, nbDesc);
}

} // namespace rsfm
