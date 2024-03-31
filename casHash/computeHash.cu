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
#include <cuda_utils.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <KArray.h>
#include "../immaUtils.cuh"
#include "../bruteForceMatch.cuh"
#include <ptr.h>
#include <ldg.cuh>

#define TRAITS_IMPORT(x) static constexpr decltype(Traits::x) x = Traits::x

#define USE_DOUBLE_BUFFER 1

namespace rsfm::cashash
{
template <typename T, typename OrigType = T>
using Ptr = cudapp::Ptr<T, OrigType>;

// First do a 64x196 MMA. 192 is 8x8(primary)+128(secondary). The original is 6x8 for primary but 176 is tricky

template <DescElemType descElemType, size_t descDims, DistanceStyle distStyle, int32_t nbChannels>
struct HashTask {
    using Descriptor = typename DescTraits<descElemType, descDims, distStyle>::Descriptor;
    using Distance = typename DescTraits<descElemType, descDims, distStyle>::Distance;
    const Descriptor* __restrict__ descA; // weights
	const int32_t* __restrict__ biasA; // bias
    static constexpr uint32_t nbDescA = nbChannels;

    const Descriptor* __restrict__ descB; // descriptors
    uint32_t nbDescB;

    KArray<uint8_t, 8>* __restrict__ bucketIndices; // [out] bucket indices for descB. The original paper uses 6 but we use 8.
	KArray<uint32_t, 4>* __restrict__ remap;
	KArray<uint32_t, 8, 256>* __restrict__ histogram; // number of elements per bucket
};

// constexpr DescElemType descElemType_ = DescElemType::kI8;
// constexpr size_t descDims_ = 128;
// constexpr DistanceStyle distStyle_ = DistanceStyle::kDotProd;
// constexpr int mmaM_ = 16;
// constexpr int mmaN_ = 8;
// constexpr int mmaK_ = 32;
template<DescElemType descElemType_, size_t descDims_, DistanceStyle distStyle_, bool bidirectional_, int mmaM_, int mmaN_, int mmaK_, int ctaWarpsH, int ctaWarpsW, int nbTilesPerCta_, bool buildHisto_>
struct MatchTraits : DescTraits<descElemType_, descDims_, distStyle_>{
    using Base = DescTraits<descElemType_, descDims_, distStyle_>;
    static constexpr DistanceStyle distStyle = Base::distStyle;
    static constexpr DescElemType descElemType = Base::descElemType;
    static constexpr int descDims = Base::descDims;
    using Distance = typename Base::Distance;
    using Word = typename Base::Word;
    static constexpr bool isDistNonNegative = true;
    static constexpr int descWords = Base::descWords;
    using Descriptor = typename Base::Descriptor;
    static constexpr HW ctaWarps = {ctaWarpsH, ctaWarpsW};
    static constexpr int warp_size = imma::warp_size; // warpSize is already defined and it's not constexpr
    static constexpr int nbBanks = imma::nbBanks;
    static constexpr HW warpShape = {8, 4};
#if !(defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ == 11 && (__CUDACC_VER_MINOR__ >= 3 && __CUDACC_VER_MINOR__ <= 8))
    // cuda-11.3/11.4/11.5/11.6/11.7/11.8 do not consider warpShape.w/.w constexpr
    static_assert(warp_size == warpShape.w * warpShape.h);
#endif
    static constexpr int ctaSize = warp_size * ctaWarps.w * ctaWarps.h;
    static constexpr int mmaM = mmaM_;
    static constexpr int mmaN = mmaN_;
    static constexpr int mmaK = mmaK_;

    using MMAType = imma::MMA<descElemType, mmaM, mmaN, mmaK>;

    // MNK config for 1 register for A, 1 register for B and two register for accumulators
    static constexpr int baseMmaM = MMAType::baseMmaM;
    static constexpr int baseMmaN = MMAType::baseMmaN;
    static constexpr int baseMmaK = MMAType::baseMmaK;

    static constexpr HW warpTileAtom = {mmaM, mmaN};
    static constexpr HW warpTileUnroll = {4, 4};
    static constexpr HW warpTile = warpTileAtom * warpTileUnroll;

    static constexpr HW ctaTileBase = warpTile * ctaWarps;
	static constexpr int nbTilesPerCta = nbTilesPerCta_;
	static constexpr HW ctaTileExt = {ctaTileBase.h, ctaTileBase.w * nbTilesPerCta};

	// should just use nbChannels = ctaTileBase.h but nvcc does not allow that, likely a bug.
	static constexpr int nbChannels = mmaM * warpTileUnroll.h * ctaWarps.h; static_assert(nbChannels == ctaTileBase.h);
	static constexpr bool buildHisto = buildHisto_;
    using Task = HashTask<descElemType, descDims, distStyle, nbChannels>;

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
		const auto laneId = int(lane_id());
        return {laneId / warpShape.w, laneId % warpShape.w};
    }
    __device__ __forceinline__
    static HW getCtaLocInGrid() {
        return {int(blockIdx.y), int (blockIdx.x)};
    }
};

struct TraitsBucketRemap : MatchTraits<DescElemType::kI8, 128, DistanceStyle::kDotProd, true, 16, 8, 32, 3, 2, 4, false>{};
struct TraitsBucketRemapHisto : MatchTraits<DescElemType::kI8, 128, DistanceStyle::kDotProd, true, 16, 8, 32, 3, 2, 4, true>{};
struct TraitsBucket : MatchTraits<DescElemType::kI8, 128, DistanceStyle::kDotProd, true, 16, 8, 32, 1, 6, 1, false>{};
struct TraitsBucketHisto : MatchTraits<DescElemType::kI8, 128, DistanceStyle::kDotProd, true, 16, 8, 32, 1, 6, 1, true>{};

// using Traits = TraitsBucketAndRemap;
template <typename Traits>
struct WarpTask : public Traits
{
    using SMemInTileA = imma::SMemInTile<Traits::descElemType, Traits::descDims, Traits::distStyle, Traits::ctaTileBase.h>;
    using SMemInTileB = imma::SMemInTile<Traits::descElemType, Traits::descDims, Traits::distStyle, Traits::ctaTileBase.w>;

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
    WarpTask(SMemInTileA& smemA_, SMemInTileB& smemB_)
        : smemA{smemA_}
        , smemB{smemB_}
    {}
	__device__ __forceinline__
	void initAcc(const Ptr<const typename Traits::Word>& ctaBias) {
        const HW warpLocInCta = Traits::getWarpLocInCta();
        const auto warpBias = ctaBias + warpTile.h * warpLocInCta.h;
        #pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
			#pragma unroll
			for (int i = 0; i < accAtomShape.h; i++) {
				#pragma unroll
				for(int w = 0; w < warpTileUnroll.w; w++) {
                    #pragma unroll
                    for (int j = 0; j < accAtomShape.w; j++) {
                        acc[h][w][i][j] = warpBias[getCoordInWarp(HW{h, w}, HW{i, j}).h];
                    }
                }
            }
        }
	}

    const SMemInTileA& smemA;
    const SMemInTileB& smemB;
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
	KArray<uint32_t, Traits::warpTile.h / 32u, Traits::warpTile.w / Traits::warp_size> getBinaryCode() {
		KArray<uint32_t, Traits::warpTile.h / 32u, Traits::warpTile.w / Traits::warp_size> results;
		static_assert(Traits::warpTile.h == 64 && Traits::warpTile.w == 32); // implemented only for this case.
		uint8_t dst[8];
        #pragma unroll
        for (int h = 0; h < warpTileUnroll.h; h++) {
			#pragma unroll
			for(int w = 0; w < warpTileUnroll.w; w++) {
				#pragma unroll
				for (int i = 0; i < accAtomShape.h; i++) {
                    #pragma unroll
                    for (int j = 0; j < accAtomShape.w; j++) {
						// transpose by shuffle then vote.
						const auto lane = lane_id();
						const auto idxTrans = lane % Traits::warpShape.h * Traits::warpShape.w + lane / Traits::warpShape.h;
						const int32_t val = __shfl_sync(~0u, acc[h][w][i][j], idxTrans);
                        const uint32_t votes = __ballot_sync(~0u, val >= 0);
						const auto laneBase = accAtomShape.w * warpShape.w * w;
						if (lane % accAtomShape.w == j && inRange(lane, laneBase, laneBase + accAtomShape.w * warpShape.w))
						{
							const auto idxByte = (lane - laneBase) / 2;
							dst[accAtomShape.h * h + i] = ((votes >> 8 * idxByte) & 0xFFu);
						}
                    }
                }
            }
        }
		static_assert(sizeof(dst) == sizeof(results));
		memcpy(&results, dst, sizeof(results));
		return results;
	}
};

// using Traits = MatchTraits;
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

	using StgBuf = KArray<uint32_t, Traits::ctaTileBase.w, 2>;

    __device__ __forceinline__
    CtaTask(const Task& task, SMemInTileA& a, SMemInTileB& b, StgBuf& stgBuf_)
		: task{task}
		, smemA{a}
		, smemB{b}
		, stgBuf{stgBuf_}
		, warpTask(a, b)
    {
    }
    const Task& task;
    SMemInTileA& smemA;
    SMemInTileB& smemB;
	StgBuf& stgBuf;
    WarpTask<Traits> warpTask;

	__device__ __forceinline__
	void loadDescA() {
		const auto ctaDescA = cudapp::Ptr<const Descriptor>(task.descA, task.nbDescA, 0);
		smemA.ctaFill<ctaSize, true>(ctaDescA);
	}

    __device__ __forceinline__
    void prologue(int idxLoop) {
		warpTask.initAcc(Ptr<const typename Traits::Word>{task.biasA, task.nbDescA, 0});

		constexpr HW ctaTileBase = Traits::ctaTileBase;
        constexpr HW ctaTileExt = Traits::ctaTileExt;
        const HW idxDescBase = ctaTileExt * Traits::getCtaLocInGrid() + ctaTileBase * HW{0, idxLoop};
		assert(idxDescBase.h == 0);
        const HW idxDescEndIfFull = idxDescBase + ctaTileBase;
        const cudapp::Ptr<const Descriptor> ctaDescB {task.descB, task.nbDescB, idxDescBase.w};
		auto transformer = [](typename Traits::Word w){
			uint32_t ret = ((reinterpret_cast<const uint32_t&>(w) & ~0x01010101u) >> 1);
			return reinterpret_cast<const typename Traits::Word&>(ret);
		};
        if (idxDescEndIfFull.w <= task.nbDescB) {
            smemB.ctaFill<ctaSize, true>(ctaDescB, transformer);
        }
        else {
            smemB.ctaFill<ctaSize, false>(ctaDescB, transformer);
        }
        __syncthreads();
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
    void epilogue(int idxLoop) {
		constexpr HW ctaTileBase = Traits::ctaTileBase;
        constexpr HW ctaTileExt = Traits::ctaTileExt;
        const uint32_t idxDescBase = ctaTileExt.w * Traits::getCtaLocInGrid().w + ctaTileBase.w * idxLoop;
		// This matches WarpTask::getBinaryCode(), not warp accumulators
		const uint32_t idxDescCtaTileBase = Traits::warp_size * Traits::getWarpLocInCta().w + lane_id();
		const uint32_t idxDesc = idxDescBase + idxDescCtaTileBase;
		const bool isInRange = (idxDesc < task.nbDescB);
		const auto binaryCode = warpTask.getBinaryCode();
		static_assert(sizeof(binaryCode) == 8);

		const auto warpLocH = Traits::getWarpLocInCta().h;
		if (warpLocH == 0) {
			if (isInRange) {
				task.bucketIndices[idxDesc] = reinterpret_cast<const KArray<uint8_t, 8>&>(binaryCode);
				if constexpr(Traits::buildHisto) {
					#pragma unroll
					for (int i = 0; i < 8; i++) {
						const uint32_t j = reinterpret_cast<const KArray<uint8_t, 8>&>(binaryCode)[i];
						atomicAdd(&(task.histogram[0][i][j]), 1);
					}
				}
			}
		}
		static_assert(Traits::ctaWarps.h == 1 || Traits::ctaWarps.h == 3);
		if constexpr (Traits::ctaWarps.h != 1) {
#if 1
			// This uses more STG instructions due to inteleaving, but seems to be slightly (2.2%) faster than the other implementation.
			if (warpLocH != 0) {
				assert(warpLocH < 3);
				if (isInRange) {
					// Note that this not ideal for STG. Can be improved with shared memory, or change the output layout.
					reinterpret_cast<KArray<uint32_t, 2>&>(task.remap[idxDesc][(warpLocH-1) * 2])
						= reinterpret_cast<const KArray<uint32_t, 2>&>(binaryCode);
				}
			}
#else
			__syncthreads();
			assert(warpLocH < 3);
			if (warpLocH == 2) {
				stgBuf[idxDescCtaTileBase] = reinterpret_cast<const KArray<uint32_t, 2>&>(binaryCode);
			}
			__syncthreads();
			if (warpLocH == 1 && isInRange) {
				KArray<uint32_t, 4> remap;
				reinterpret_cast<KArray<uint32_t, 2>&>(remap[0]) = reinterpret_cast<const KArray<uint32_t, 2>&>(binaryCode);
				reinterpret_cast<KArray<uint32_t, 2>&>(remap[2]) = stgBuf[idxDescCtaTileBase];
				task.remap[idxDesc] = remap;
			}
#ifndef NDEBUG
			__syncthreads();
			if (warpLocH >= 1 && isInRange) {
				assert(reinterpret_cast<const uint64_t&>(task.remap[idxDesc][(warpLocH-1) * 2])
					== reinterpret_cast<const uint64_t&>(binaryCode));
			}
#endif
#endif
		}
    }
};


template <typename Traits>
__global__ void
#ifdef NDEBUG
__launch_bounds__(Traits::ctaSize, 2)
#endif
kernelComputeHash(const typename Traits::Task task) {
    using SMemInTileA = typename WarpTask<Traits>::SMemInTileA;
    using SMemInTileB = typename WarpTask<Traits>::SMemInTileB;

    __shared__ struct {
        typename CtaTask<Traits>::SMemInTileA inTileA;
        union {
            typename CtaTask<Traits>::SMemInTileB inTileB;
			typename CtaTask<Traits>::StgBuf stgBuf;
        };
    } smem;

    constexpr auto ctaTileExt = Traits::ctaTileExt;
    const HW idxDescBase = ctaTileExt * Traits::getCtaLocInGrid();
	assert(idxDescBase.h == 0);
    if (idxDescBase.w >= task.nbDescB) {
        return;
    }
    CtaTask<Traits> ctaTask{task, smem.inTileA, smem.inTileB, smem.stgBuf};
	ctaTask.loadDescA();
	// #pragma unroll
	for (int i = 0; i < Traits::nbTilesPerCta; i++) {
		if (idxDescBase.w + Traits::ctaTileBase.w * i >= task.nbDescB) {
			break;
		}
		ctaTask.prologue(i);
		ctaTask.mainloop();
		ctaTask.epilogue(i);
		__syncthreads();
	}
}

void computeHashForBucketAndRemap(const KArray<int32_t, 32>* __restrict__ weights, const int32_t* __restrict__ bias,
	const KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc,
    KArray<uint8_t, 8>* __restrict__ bucketIndices, KArray<uint32_t, 4>* __restrict__ remap,
	cudapp::KArray<uint32_t, 8, 256>* __restrict__ histogram, cudaStream_t stream)
{
	if (histogram != nullptr) {
		cudaCheck(cudaMemsetAsync(histogram, 0, sizeof(*histogram), stream));
	}
	const HashTask<DescElemType::kI8, 128, DistanceStyle::kDotProd, 192> task{
		weights, bias,
		reinterpret_cast<const KArray<int32_t, 32>*>(desc), nbDesc,
		bucketIndices, remap,
		histogram
	};
	if (histogram != nullptr) {
		using Traits = TraitsBucketRemapHisto;
		launchKernel(&kernelComputeHash<Traits>, divUp(task.nbDescB, uint32_t(Traits::ctaTileExt.w)), Traits::ctaSize, 0, stream, task);
	}
	else {
		using Traits = TraitsBucketRemap;
		launchKernel(&kernelComputeHash<Traits>, divUp(task.nbDescB, uint32_t(Traits::ctaTileExt.w)), Traits::ctaSize, 0, stream, task);
	}
}

void computeHashForBucketOnly(const KArray<int32_t, 32>* __restrict__ weights, const int32_t* __restrict__ bias,
	const KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc,
    KArray<uint8_t, 8>* __restrict__ bucketIndices,
	cudapp::KArray<uint32_t, 8, 256>* __restrict__ histogram,  cudaStream_t stream)
{
	if (histogram != nullptr) {
		cudaCheck(cudaMemsetAsync(histogram, 0, sizeof(*histogram), stream));
	}
	const HashTask<DescElemType::kI8, 128, DistanceStyle::kDotProd, 64> task{
		weights, bias,
		reinterpret_cast<const KArray<int32_t, 32>*>(desc), nbDesc,
		bucketIndices, nullptr,
		histogram
	};
	if (histogram != nullptr) {
		using Traits = TraitsBucketHisto;
		launchKernel(&kernelComputeHash<Traits>, divUp(task.nbDescB, uint32_t(Traits::ctaTileExt.w)), Traits::ctaSize, 0, stream, task);
	}
	else {
		using Traits = TraitsBucket;
		launchKernel(&kernelComputeHash<Traits>, divUp(task.nbDescB, uint32_t(Traits::ctaTileExt.w)), Traits::ctaSize, 0, stream, task);
	}
}

}

