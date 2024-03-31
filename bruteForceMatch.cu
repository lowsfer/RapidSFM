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

//
// Created by yao on 3/20/20.
//

#include <cuda_hint.cuh>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <cstdint>
#include <limits>
#include <array>
#include <KArray.h>
#include <cooperative_groups.h>
#include <cassert>
#include <numeric>
#include <device_launch_parameters.h>
#include <cuda_utils.h>
#include "bruteForceMatch.h"
#include <vector>
#include "bruteForceMatch.cuh"

namespace cg = cooperative_groups;

// For < sm_75, double buffer will use too many registers when using L2 norm. DotProd is fine.
// @fixme: save some registers for double buffer on sm_61
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define USE_DOUBLE_BUFFER 1
#else
#define USE_DOUBLE_BUFFER 0
#endif

namespace rsfm {

// See https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/ for task break down
template<DescElemType descElemType_, size_t descDims_, DistanceStyle distStyle_, bool bidirectional_>
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
    static constexpr int warp_size = 32; // warpSize is already defined and it's not constexpr
    static constexpr int nbBanks = 32;
    static constexpr HW warpShape = {8, 4};
#if !(defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 3 && __CUDACC_VER_MINOR__ <= 8))
    // cuda-11.3/11.4/11.5/11.6 do no consider warpShape.w/.w constexpr
    static_assert(warp_size == warpShape.w * warpShape.h);
#endif
    static constexpr int ctaSize = warp_size * ctaWarps.w * ctaWarps.h;
    static constexpr HW thrdTileAtom = {4, 4}; // used as words per SMem vector load
    static constexpr HW warpTileAtom = thrdTileAtom * warpShape;
    static constexpr HW warpTileUnroll = {2, 2};
    static constexpr HW warpTile = warpTileAtom * warpTileUnroll;
    static constexpr HW warpTileInAtoms = warpTile / thrdTileAtom;
    static constexpr int gmemLdWords = 4;

    static constexpr HW ctaTile = warpTile * ctaWarps;

    using BestMatchCta = ReducedBestMatchType<false, 8, 24, true>;
    using Index = typename BestMatchCta::Index;
    using BestMatchGlobal = BestMatchType<Distance, true, true>;

    using SMemLdVecA = KArray<Word, size_t(thrdTileAtom.h)>;
    using SMemLdVecB = KArray<Word, size_t(thrdTileAtom.w)>;
    using AccAtom = KArray<Word, size_t(thrdTileAtom.h), size_t(thrdTileAtom.w)>;

    __device__ __forceinline__
    static AccAtom computeAtom(const AccAtom& init, const SMemLdVecA& a, const SMemLdVecB& b) {
        AccAtom acc = init;
        for (int i = 0; i < thrdTileAtom.h; i++) {
            for (int j = 0; j < thrdTileAtom.w; j++) {
                acc[i][j] = accumulateDistance<descElemType, distStyle>(acc[i][j], a[i], b[j]);
            }
        }
        return acc;
    }
    static constexpr int nbRegBufs = USE_DOUBLE_BUFFER ? 2 : 1;
};

using SiftBiDirectionalTraits = MatchTraits<DescElemType::kU8, 128, DistanceStyle::kL2, true>;
// using SiftUniDirectionalTraits = MatchTraits<DescElemType::kU8, 128, DistanceStyle::kL2, false>;

using Traits = SiftBiDirectionalTraits;

/* Swizzle pattern: (in a smaller scale, with rowSliding = 1)
 * a0 b0 c0 d0 | e0 f0 g0 h0
 * d1 a1 b1 c1 | h1 e1 f1 g1
 * c2 d2 a2 b2 | g2 h2 e2 f2
 * b3 c3 d3 a3 | f3 g3 h3 e3
 * ------------|------------
 * e4 f4 d4 h4 | a4 b4 c4 d4
 * h5 e5 f5 g5 | d5 a5 b5 c5
 * g6 h6 e6 f6 | c6 d6 a6 b6
 * f7 g7 h7 e7 | b7 c7 d7 a7
 */
template <typename Word, int nbDesc, int smemLdWords, int gmemLdWords>
struct alignas(16) SMemInTile : public Traits
{
    using LdVec = KArray<Word, smemLdWords>;
    static_assert(nbBanks * gmemLdWords % descWords == 0);
#if 1
    // this has 4x bank conflict but turns out to be faster and uses less registers on sm_61, likely due to 16-byte access
    static constexpr int rowSliding = 4;
#else
    static constexpr int rowSliding = nbBanks / descWords; // This has no bank conflict but is slower on sm_61 and uses more registers
#endif
    static constexpr int atomsPerRow = nbDesc / smemLdWords;
    static constexpr int rowSlidingMinor = rowSliding % smemLdWords;
    static constexpr int nbUnroll = smemLdWords / (rowSlidingMinor == 0 ? smemLdWords : rowSlidingMinor);
    __device__ __forceinline__
    KArray<Word, smemLdWords> load(int idxWord, int idxLdVec /* = (idxDescBegInCta / LdNbDesc) */, int offsetInsideAtom) const {
        assert(idxWord * rowSliding % smemLdWords == offsetInsideAtom); // we pass offsetInsideAtom in to help compiler unroll
        const LdVec atom = data[idxWord][(idxLdVec + idxWord * rowSliding / smemLdWords) % atomsPerRow];
        LdVec result;
#pragma unroll
        for (int i = 0; i < smemLdWords; i++){
            result[i] = atom[(offsetInsideAtom + i) % smemLdWords];
        }
        return result;
    }
    __device__ __forceinline__
    void store(int idxDescInCta, int idxWord, Word val) {
        LdVec& atom = data[idxWord][(idxDescInCta / smemLdWords + idxWord * rowSliding / smemLdWords) % atomsPerRow];
        atom[(idxWord * rowSliding + idxDescInCta) % smemLdWords] = val;
    }

    // desc is the offseted pointer for this tile, and nbLdDesc is the actually descriptors to load
    __device__ __forceinline__
    void fill(const cg::thread_block& cta, const Descriptor* desc, int nbLdDesc) {
        assert(nbLdDesc <= nbDesc);
        using GMemLdVec = KArray<Word, gmemLdWords>;
        static constexpr int descGMemVecs = descWords / gmemLdWords;
        static_assert(descWords % gmemLdWords == 0);
        using DescVecs = KArray<GMemLdVec, descGMemVecs>;

        const int tid = int(cta.thread_rank());
        constexpr int nbLdVecsA = descWords * ctaTile.h / gmemLdWords;
        static_assert(sizeof(GMemLdVec) * nbLdVecsA == sizeof(*this));
        #pragma unroll
        for (int iter = 0; iter < divUp(nbLdVecsA, ctaSize); iter++) {
            const int idxVecCta = ctaSize * iter + tid;
            const int idxDescCta = idxVecCta / descGMemVecs;
            const int idxVecInsideDesc = idxVecCta % descGMemVecs;
            GMemLdVec vec;
            if (idxDescCta < nbLdDesc) {
                vec = reinterpret_cast<const DescVecs&>(desc[idxDescCta])[idxVecInsideDesc];
            }
            #pragma unroll
            for (int i = 0; i < gmemLdWords; i++) {
                this->store(idxDescCta, gmemLdWords * idxVecInsideDesc + i, vec[i]);
            }
        }
    }

    KArray<LdVec, descWords, atomsPerRow> data;
};

//template <typename Traits = SiftTraits>
struct ThrdTask : public Traits
{
    KArray<SMemLdVecA, warpTileUnroll.h> a[nbRegBufs];
    KArray<SMemLdVecB, warpTileUnroll.w> b[nbRegBufs];
    KArray<AccAtom, warpTileUnroll.h, warpTileUnroll.w> acc{};

    __device__ __forceinline__
    void compute(int idxBuf, int i, int j) {
        acc[i][j] = computeAtom(acc[i][j], a[idxBuf][i], b[idxBuf][j]);
    }

    __device__ __forceinline__
    void disgardOutOfBoundValues(const HW idxDescBegInsideCta, const HW realCtaTile) {
        if (realCtaTile.h == ctaTile.h && realCtaTile.w == ctaTile.w) {
            return;
        }
        if (realCtaTile.h < ctaTile.h)
        {
#pragma unroll
            for (int i = 0; i < warpTileUnroll.h; i++) {
#pragma unroll
                for (int k = 0; k < thrdTileAtom.h; k++) {
                    const Index idxAInCta = idxDescBegInsideCta.h + warpTileAtom.h * i + k;
                    if (idxAInCta >= realCtaTile.h) {
#pragma unroll
                        for (int j = 0; j < warpTileUnroll.w; j++) {
#pragma unroll
                            for (int l = 0; l < thrdTileAtom.w; l++) {
                                acc[i][j][k][l] = BestMatchCta::worstValue;
                            }
                        }
                    }
                }
            }
        }
        if (realCtaTile.w < ctaTile.w)
        {
#pragma unroll
            for (int j = 0; j < warpTileUnroll.w; j++) {
#pragma unroll
                for (int l = 0; l < thrdTileAtom.w; l++) {
                    const Index idxBInCta = idxDescBegInsideCta.w + warpTileAtom.w * j + l;
                    if (idxBInCta >= realCtaTile.w) {
#pragma unroll
                        for (int i = 0; i < warpTileUnroll.h; i++) {
#pragma unroll
                            for (int k = 0; k < thrdTileAtom.h; k++) {
                                acc[i][j][k][l] = BestMatchCta::worstValue;
                            }
                        }
                    }
                }
            }
        }
    }

    // indices are local indices in CTA, i.e. in range [0, ctaTile)
    __device__ __forceinline__
    std::pair<KArray<BestMatchCta, warpTileUnroll.h, thrdTileAtom.h>, KArray<BestMatchCta, warpTileUnroll.w, thrdTileAtom.w>>
    computeThrdBestMatches(const HW idxDescBegInsideCta) {
        KArray<BestMatchCta, warpTileUnroll.h, thrdTileAtom.h> bestMatchA;
        KArray<BestMatchCta, warpTileUnroll.w, thrdTileAtom.w> bestMatchB;
#pragma unroll
        for (int i = 0; i < warpTileUnroll.h; i++) {
#pragma unroll
            for (int j = 0; j < warpTileUnroll.w; j++) {
#pragma unroll
                for (int k = 0; k < thrdTileAtom.h; k++) {
#pragma unroll
                    for (int l = 0; l < thrdTileAtom.w; l++) {
                        const Distance distance = acc[i][j][k][l];
                        {
                            const Index idxBInCta = idxDescBegInsideCta.w + warpTileAtom.w * j + l;
                            const BestMatchCta m = BestMatchCta::makeInstance(distance, idxBInCta);
                            if (j == 0 && l == 0) {
                                bestMatchA[i][k] = m;
                            }
                            else {
                                bestMatchA[i][k].updateRegister(m);
                            }
                        }
                        if (bidirectional)
                        {
                            const Index idxAInCta = idxDescBegInsideCta.h + warpTileAtom.h * i + k;
                            const BestMatchCta m = BestMatchCta::makeInstance(distance, idxAInCta);
                            if (i == 0 && k == 0) {
                                bestMatchB[j][l] = m;
                            }
                            else {
                                bestMatchB[j][l].updateRegister(m);
                            }
                        }
                    }
                }
            }
        }
        return std::make_pair(bestMatchA, bestMatchB);
    }
};

//template <typename Traits = SiftTraits>
struct WarpTask : public Traits
{
    using Warp = cg::thread_block_tile<warp_size>;

    using SMemInTileA = SMemInTile<Word, ctaTile.h, thrdTileAtom.h, gmemLdWords>;
    using SMemInTileB = SMemInTile<Word, ctaTile.w, thrdTileAtom.w, gmemLdWords>;

    __device__ __forceinline__
    WarpTask(const Warp& warp, SMemInTileA& a, SMemInTileB& b, int idxWarp)
    : warp{warp}
    , smemA{a}
    , smemB{b}
    , warpLocInCta{idxWarp / ctaWarps.w, idxWarp % ctaWarps.w}
    , thrdLocInWarp{int(warp.thread_rank()) / warpShape.w, int(warp.thread_rank()) % warpShape.w}
    {
    }
    const Warp warp;
    const SMemInTileA& smemA;
    const SMemInTileB& smemB;
    const HW warpLocInCta;
    const HW thrdLocInWarp; // {lane_id() / warpShape.w, land_id() % warpShape.w}
    ThrdTask thrdTask;

    __device__ __forceinline__
    void computeStep(int idxBuf) {
        for (int i = 0; i < warpTileUnroll.h; i++) {
            for (int j = 0; j < warpTileUnroll.w; j++) {
                thrdTask.compute(idxBuf, i, j);
            }
        }
    }
    __device__ __forceinline__
    void smemLdStep(int idxWord, int idxBuf /* = idxWord % 2 */, int offsetInsideLdVecA, int offsetInsideLdVecB) {
        assert(idxBuf == idxWord % nbRegBufs);
        assert(idxWord * SMemInTileA::rowSliding % thrdTileAtom.h == offsetInsideLdVecA);
        assert(idxWord * SMemInTileB::rowSliding % thrdTileAtom.w == offsetInsideLdVecB);
        constexpr HW warpTileInAtoms = Traits::warpTileInAtoms;
        const HW warpTileOffsetInAtoms = warpTileInAtoms * warpLocInCta;
        for (int i = 0; i < warpTileUnroll.h; i++) {
            thrdTask.a[idxBuf][i] = smemA.load(idxWord, warpTileOffsetInAtoms.h + warpShape.h * i + thrdLocInWarp.h, offsetInsideLdVecA);
        }
        for (int j = 0; j < warpTileUnroll.w; j++) {
            thrdTask.b[idxBuf][j] = smemB.load(idxWord, warpTileOffsetInAtoms.w + warpShape.w * j + thrdLocInWarp.w, offsetInsideLdVecB);
        }
    }

#if USE_DOUBLE_BUFFER
    __device__ __forceinline__
    void compute(){
        static_assert(nbRegBufs == 2);
        smemLdStep(0, 0, 0, 0);
#if 0
        constexpr int nbUnroll = std::lcm(SMemInTileA::nbUnroll, SMemInTileB::nbUnroll);
#else
        constexpr int nbUnroll = std::max(nbRegBufs, std::max(SMemInTileA::nbUnroll, SMemInTileB::nbUnroll));
        static_assert(nbUnroll % SMemInTileA::nbUnroll == 0 && nbUnroll % SMemInTileB::nbUnroll == 0);
#endif
        static_assert(nbUnroll > 0 && descWords % nbUnroll == 0);

#define SPECIALIZE_LAST_ITERATION 1 // @fixme: check which is better
#if SPECIALIZE_LAST_ITERATION
        auto innerLoop = [&](int iterOuter, bool isLastOuterIter){
#pragma unroll
            for (int iterInner = 0; iterInner < nbUnroll; iterInner++) {
                const int idxWord = nbUnroll * iterOuter + iterInner;
                if (!(isLastOuterIter && iterInner == nbUnroll - 1)) {
                    smemLdStep(idxWord + 1, (iterInner + 1) % nbRegBufs,
                            SMemInTileA::rowSlidingMinor * (iterInner + 1) % thrdTileAtom.h,
                            SMemInTileB::rowSlidingMinor * (iterInner + 1) % thrdTileAtom.w);
                }
                computeStep(iterInner % nbRegBufs);
            }
        };
        static_assert(descWords / nbUnroll >= 1);
#pragma unroll(1)
        for (int iterOuter = 0; iterOuter < descWords / nbUnroll - 1; iterOuter++) {
            innerLoop(iterOuter, false);
        }
        innerLoop(descWords / nbUnroll - 1, true);
#else
#pragma unroll(1)
        for (int iterOuter = 0; iterOuter < descWords / nbUnroll; iterOuter++) {
#pragma unroll
            for (int iterInner = 0; iterInner < nbUnroll; iterInner++) {
                const int idxWord = nbUnroll * iterOuter + iterInner;
                if (iterInner == nbUnroll - 1 && idxWord < descWords - 1) {
                    smemLdStep(idxWord + 1, (iterInner + 1) % nbRegBufs,
                            SMemInTileA::rowSlidingMinor * (iterInner + 1) % thrdTileAtom.h,
                            SMemInTileB::rowSlidingMinor * (iterInner + 1) % thrdTileAtom.w);
                }
                computeStep(iterInner % nbRegBufs);
            }
        }
#endif
    }
#else
    __device__ __forceinline__
    void compute(){
        static_assert(nbRegBufs == 1);
#if 0
        constexpr int nbUnroll = std::lcm(SMemInTileA::nbUnroll, SMemInTileB::nbUnroll);
#else
        constexpr int nbUnroll = std::max(SMemInTileA::nbUnroll, SMemInTileB::nbUnroll);
        static_assert(nbUnroll % SMemInTileA::nbUnroll == 0 && nbUnroll % SMemInTileB::nbUnroll == 0);
#endif
        static_assert(nbUnroll > 0 && descWords % nbUnroll == 0);
#pragma unroll(1)
        for (int iterOuter = 0; iterOuter < descWords / nbUnroll; iterOuter++) {
#pragma unroll
            for (int iterInner = 0; iterInner < nbUnroll; iterInner++) {
                const int idxWord = nbUnroll * iterOuter + iterInner;
                smemLdStep(idxWord, iterInner % nbRegBufs,
                        SMemInTileA::rowSlidingMinor * iterInner % thrdTileAtom.h,
                        SMemInTileB::rowSlidingMinor * iterInner % thrdTileAtom.w);
                computeStep(iterInner % nbRegBufs);
            }
        }
    }
#endif

    __device__ __forceinline__
    void disgardOutOfBoundValues(const HW realCtaTile) {
        constexpr HW warpTile = Traits::warpTile;
        constexpr HW thrdTileAtom = Traits::thrdTileAtom;
        const HW idxDescBegInsideCta = warpTile * warpLocInCta + thrdTileAtom * thrdLocInWarp;
        thrdTask.disgardOutOfBoundValues(idxDescBegInsideCta, realCtaTile);
    }

    __device__ __forceinline__
    std::pair<KArray<BestMatchCta, warpTileUnroll.h, thrdTileAtom.h>, KArray<BestMatchCta, warpTileUnroll.w, thrdTileAtom.w>>
    computeWarpBestMatches() {
        constexpr HW warpTile = Traits::warpTile;
        constexpr HW thrdTileAtom = Traits::thrdTileAtom;
        const HW idxDescBegInsideCta = warpTile * warpLocInCta + thrdTileAtom * thrdLocInWarp;
        const auto thrdBestMatches = thrdTask.computeThrdBestMatches(idxDescBegInsideCta);
        KArray<BestMatchCta, warpTileUnroll.h, thrdTileAtom.h> bestMatchA = thrdBestMatches.first;
        KArray<BestMatchCta, warpTileUnroll.w, thrdTileAtom.w> bestMatchB = thrdBestMatches.second;
#pragma unroll
        for (int i = 0; i < warpTileUnroll.h; i++) {
#pragma unroll
            for (int j = 0; j < thrdTileAtom.h; j++) {
                BestMatchCta& m = bestMatchA[i][j];
#pragma unroll
                for (int k = warpShape.w / 2; k > 0; k /= 2){
                    const uint32_t otherRaw = warp.shfl_xor(m.raw, k);
                    m.updateRegister(otherRaw);
                }
            }
        }
        if (bidirectional) {
#pragma unroll
            for (int i = 0; i < warpTileUnroll.w; i++) {
#pragma unroll
                for (int j = 0; j < thrdTileAtom.w; j++) {
                    BestMatchCta& m = bestMatchB[i][j];
#pragma unroll
                    for (int k = warp_size / 2; k >= warpShape.w; k /= 2) {
                        const uint32_t otherRaw = warp.shfl_xor(m.raw, k);
                        m.updateRegister(otherRaw);
                    }
                }
            }
        }
        return std::make_pair(bestMatchA, bestMatchB);
    }
};

struct CtaTask : Traits
{
    using SMemInTileA = SMemInTile<Word, ctaTile.h, thrdTileAtom.h, gmemLdWords>;
    using SMemInTileB = SMemInTile<Word, ctaTile.w, thrdTileAtom.w, gmemLdWords>;

    __device__ __forceinline__
    CtaTask(const cg::thread_block& cta, const Task& task, SMemInTileA& a, SMemInTileB& b,
        KArray<typename Traits::BestMatchCta, Traits::ctaWarps.w, Traits::ctaTile.h>& smemBestMatchA,
        KArray<typename Traits::BestMatchCta, Traits::ctaWarps.h, Traits::ctaTile.w>& smemBestMatchB)
    : cta{cta}
    , ctaLocInGrid{int(cta.group_index().y), int (cta.group_index().x)}
    , warp{cg::tiled_partition<warp_size>(cta)}
    , task{task}
    , smemA{a}
    , smemB{b}
    , warpTask{warp, a, b, int(cta.thread_rank()) / warp_size}
    , smemBestMatchA(smemBestMatchA)
    , smemBestMatchB(smemBestMatchB)
    {
    }
    const cg::thread_block cta;
    const HW ctaLocInGrid;
    const cg::thread_block_tile<warp_size> warp;
    const Task& task;
    SMemInTileA& smemA;
    SMemInTileB& smemB;
    WarpTask warpTask;
    KArray<typename Traits::BestMatchCta, ctaWarps.w, ctaTile.h>& smemBestMatchA;
    KArray<typename Traits::BestMatchCta, ctaWarps.h, ctaTile.w>& smemBestMatchB;

    __device__ __forceinline__
    void prologue() {
        constexpr HW ctaTile = Traits::ctaTile;
        const HW idxDescBase = ctaTile * ctaLocInGrid;
        const HW realCtaTile = {
            std::min(ctaTile.h, int(task.nbDescA) - idxDescBase.h),
            std::min(ctaTile.w, int(task.nbDescB) - idxDescBase.w)
        };
        smemA.fill(cta, task.descA + idxDescBase.h, realCtaTile.h);
        smemB.fill(cta, task.descB + idxDescBase.w, realCtaTile.w);
        cta.sync();
    }
    __device__ __forceinline__
    void mainloop() {
        warpTask.compute();
    }
    __device__ __forceinline__
    void epilogue(){
        constexpr HW ctaTile = Traits::ctaTile;
        const HW idxDescBase = ctaTile * ctaLocInGrid;
        const HW realCtaTile = {
            std::min(ctaTile.h, int(task.nbDescA) - idxDescBase.h),
            std::min(ctaTile.w, int(task.nbDescB) - idxDescBase.w)
        };
        warpTask.disgardOutOfBoundValues(realCtaTile);
        const auto warpBestMatches = warpTask.computeWarpBestMatches();
        const auto& warpBestMatchA = warpBestMatches.first;
        const auto& warpBestMatchB = warpBestMatches.second;
        const auto warpLocInCta = warpTask.warpLocInCta;
        cta.sync(); // required because smemBestMatchA/B overlaps input buffer
        if (warp.thread_rank() % warpShape.w == 0) {
            for (int i = 0; i < warpTileUnroll.h; i++) {
                using BestMatchAtomA = KArray<BestMatchCta, thrdTileAtom.h>;
                // nvvp think there is bank conflict here. I printed address out and there is no conflict.
                reinterpret_cast<BestMatchAtomA&>(smemBestMatchA[warpLocInCta.w][warpTile.h * warpLocInCta.h + warpTileAtom.h * i + thrdTileAtom.h * (warp.thread_rank()/warpShape.w)]) = warpBestMatchA[i];
            }
        }
        if (Traits::bidirectional && warp.thread_rank() / warpShape.w == 0) {
            for (int i = 0; i < warpTileUnroll.w; i++) {
                using BestMatchAtomB = KArray<BestMatchCta, thrdTileAtom.w>;
                reinterpret_cast<BestMatchAtomB&>(smemBestMatchB[warpLocInCta.h][warpTile.w * warpLocInCta.w + warpTileAtom.w * i + thrdTileAtom.w * warp.thread_rank()]) = warpBestMatchB[i];
            }
        }
        cta.sync();

        const int idxDescBaseA = ctaTile.h * ctaLocInGrid.h;
        const int idxDescBaseB = ctaTile.w * ctaLocInGrid.w;
        for (int n = 0; n < divUp(ctaTile.h, ctaSize); n++) {
            const int idxDescCta = ctaSize * n + cta.thread_rank();
            if (idxDescCta >= ctaTile.h) {
                break;
            }
            BestMatchCta ctaBestMatchA = smemBestMatchA[0][idxDescCta];
            for (int i = 1; i < ctaWarps.w; i++) {
                ctaBestMatchA.updateRegister(smemBestMatchA[i][idxDescCta].raw);
            }
            if (idxDescBaseA + idxDescCta < task.nbDescA) {
                reinterpret_cast<BestMatchGlobal&>(task.bestMatchA[idxDescBaseA + idxDescCta]).updateRam(ctaBestMatchA.distance(), ctaBestMatchA.index() + idxDescBaseB);
            }
        }
        if (Traits::bidirectional) {
            for (int n = 0; n < divUp(ctaTile.w, ctaSize); n++) {
                const int idxDescCta = ctaSize * n + cta.thread_rank();
                if (idxDescCta >= ctaTile.w) {
                    break;
                }
                BestMatchCta ctaBestMatchB = smemBestMatchB[0][idxDescCta];
                for (int i = 1; i < ctaWarps.h; i++) {
                    ctaBestMatchB.updateRegister(smemBestMatchB[i][idxDescCta].raw);
                }
                if (idxDescBaseB + idxDescCta < task.nbDescB) {
                    reinterpret_cast<BestMatchGlobal&>(task.bestMatchB[idxDescBaseB + idxDescCta]).updateRam(ctaBestMatchB.distance(), ctaBestMatchB.index() + idxDescBaseA);
                }
            }
        }
    }
};

static constexpr size_t maxNbTasks = 16;

__global__ void
__launch_bounds__(Traits::ctaSize)
kernelBruteForceMatch(const std::array<typename Traits::Task, maxNbTasks> tasks) {
    __shared__ struct {
        CtaTask::SMemInTileA inTileA;
        union {
            CtaTask::SMemInTileB inTileB;
            struct {
                KArray<typename Traits::BestMatchCta, Traits::ctaWarps.w, Traits::ctaTile.h> bestMatchA;
                KArray<typename Traits::BestMatchCta, Traits::ctaWarps.h, Traits::ctaTile.w> bestMatchB;
            };
        };
    } smem;
    const auto cta = cg::this_thread_block();
    const auto idxTask = cta.group_index().z;
    assert(idxTask < tasks.size());
    const auto& task = tasks[idxTask];
    const HW ctaLocInGrid{int(cta.group_index().y), int(cta.group_index().x)};
    constexpr auto ctaTile = Traits::ctaTile;
    const HW idxDescBase = ctaTile * ctaLocInGrid;
    if (idxDescBase.h >= task.nbDescA || idxDescBase.w >= task.nbDescB) {
        return;
    }
    CtaTask ctaTask{cta, task, smem.inTileA, smem.inTileB, smem.bestMatchA, smem.bestMatchB};
    ctaTask.prologue();
    ctaTask.mainloop();
    ctaTask.epilogue();
}

static constexpr uint32_t ctaSizeForInit = 256;

void launchBruteForceMatchImpl(const typename Traits::Task* tasks, size_t nbTasks, cudaStream_t stream)
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
    launchKernel(&kernelBruteForceMatch, gridForMatch, Traits::ctaSize, 0, stream, argTasks);
}

void launchBruteForceMatchIDP(const typename Traits::Task* tasks, size_t nbTasks, cudaStream_t stream)
{
    for (size_t i = 0; i < nbTasks; i += maxNbTasks) {
        launchBruteForceMatchImpl(tasks + i, std::min(maxNbTasks, nbTasks - i), stream);
    }
}

}

