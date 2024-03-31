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
#include "cascadeHash.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace rsfm::cashash
{

static constexpr uint32_t ctaSize = 256;
static constexpr uint32_t unroll = 16;
using LdVec = cudapp::KArray<uint32_t, 4>;
static_assert(alignof(LdVec) == 16);
using Descriptor = cudapp::KArray<uint32_t, 32>;
static constexpr uint32_t grpSize = sizeof(Descriptor) / sizeof(LdVec);

template <bool accSqr>
__global__ void kernelAccumulateDesc(cudapp::KArray<uint64_t, 128>& __restrict__ acc, cudapp::KArray<uint64_t, 128>* __restrict__ pSqrAcc, const Descriptor* __restrict__ desc, uint32_t nbDesc) {
	__shared__ cudapp::KArray<uint32_t, 128> ctaAcc_;
	__shared__ cudapp::KArray<uint32_t, 128> ctaSqrAcc_;
	static_assert(ctaSize >= 128);
	if (threadIdx.x < 128) {
		ctaAcc_[threadIdx.x] = 0u;
		if constexpr (accSqr) {
			ctaSqrAcc_[threadIdx.x] = 0u;
		}
	}
	__syncthreads();

	auto ctaAccAt = [&](uint32_t i) -> uint32_t& {
		const uint32_t h = i / 16;
		const uint32_t w = i % 16;
		return ctaAcc_[h * 16 + (h + w)%16];
	};
	auto ctaSqrAccAt = [&](uint32_t i) -> uint32_t& {
		const uint32_t h = i / 16;
		const uint32_t w = i % 16;
		return ctaSqrAcc_[h * 16 + (h + w)%16];
	};

	const uint32_t idxGrp = (ctaSize * blockIdx.x + threadIdx.x) / grpSize;
	const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
	const uint32_t lane = g.thread_rank();

	cudapp::KArray<uint32_t, 16> thrdAcc{};
	cudapp::KArray<uint32_t, 16> thrdSqrAcc{};
	
	const uint32_t idxDescBase = unroll * idxGrp;
	#pragma unroll
	for (uint32_t i = 0; i < unroll; i++) {
		const uint32_t idxDesc = idxDescBase + i;
		LdVec srcRep = idxDesc < nbDesc ? reinterpret_cast<const cudapp::KArray<LdVec, grpSize>&>(desc[idxDescBase + i])[lane] : LdVec{};
		for (int j = 0; j < 4; j++) {
			srcRep[j] = ((srcRep[j] & ~0x01010101u) >> 1); // same as computeHash transformation.
		}
		const auto& src = reinterpret_cast<const cudapp::KArray<uint8_t, 16>&>(srcRep);
		#pragma unroll
		for (uint32_t j = 0; j < 16; j++) {
			thrdAcc[j] += src[j];
			if constexpr (accSqr) {
				thrdSqrAcc[j] += uint32_t(src[j]) * src[j];
			}
		}
	}
#if 0 // slower with this. though bank conflict is avoided.
	__syncwarp();
	#pragma unroll
	for (uint32_t xorMask = 16; xorMask >= grpSize; xorMask /= 2) {
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			uint32_t other = __shfl_xor_sync(~0U, thrdAcc[i], xorMask);
			thrdAcc[i] += other;
			if constexpr (accSqr) {
				other = __shfl_xor_sync(~0U, thrdSqrAcc[i], xorMask);
				thrdSqrAcc[i] += other;
			}
		}
	}
	if (lane_id() < grpSize)
#endif
	{
		#pragma unroll
		for (int i = 0; i < 16; i++) {
			atomicAdd_block(&ctaAccAt(16 * lane + i), thrdAcc[i]);
			if constexpr (accSqr) {
				atomicAdd_block(&ctaSqrAccAt(16 * lane + i), thrdSqrAcc[i]);
			}
		}
	}
	__syncthreads();

	static_assert(ctaSize >= 128);
	if (threadIdx.x < 128) {
		atomicAdd(reinterpret_cast<unsigned long long*>(&acc[threadIdx.x]), static_cast<unsigned long long>(ctaAccAt(threadIdx.x)));
		if constexpr(accSqr) {
			atomicAdd(reinterpret_cast<unsigned long long*>(&pSqrAcc[0][threadIdx.x]), static_cast<unsigned long long>(ctaSqrAccAt(threadIdx.x)));
		}
	}
}

void accumuateDesc(cudapp::KArray<uint64_t, 128>& acc, cudapp::KArray<uint64_t, 128>* __restrict__ sqrAcc, bool initAcc, const cudapp::KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc, cudaStream_t stream) {
	if (initAcc) {
		cudaCheck(cudaMemsetAsync(&acc, 0, sizeof(acc), stream));
		if (sqrAcc) {
			cudaCheck(cudaMemsetAsync(sqrAcc, 0, sizeof(*sqrAcc), stream));
		}
	}
	if (sqrAcc == nullptr) {
		kernelAccumulateDesc<false><<<divUp(nbDesc, ctaSize / grpSize * unroll), ctaSize, 0, stream>>>(acc, sqrAcc, desc, nbDesc);
	}
	else {
		kernelAccumulateDesc<true><<<divUp(nbDesc, ctaSize / grpSize * unroll), ctaSize, 0, stream>>>(acc, sqrAcc, desc, nbDesc);
	}
	cudaCheck(cudaGetLastError());
}


static constexpr uint32_t ctaSize_computeBias = 512;

__global__ void kernelComputeBias(cudapp::KArray<int32_t, 64>& __restrict__ bias,
	const cudapp::KArray<int32_t, 64, 32>& __restrict__ weights, const cudapp::KArray<int32_t, 64>& __restrict__ biasBase,
	const cudapp::KArray<uint64_t, 128>& __restrict__ acc, uint64_t nbDesc)
{
	const auto tid = threadIdx.x;
	__shared__ cudapp::KArray<float, 4, 32> negCenter_;
	auto negCenter = [&](uint32_t idx) -> float& {
		const uint32_t i = idx / 32;
		const uint32_t j = idx % 32;
		return negCenter_[i][(j + i * 4) % 32];
	};
	assert(blockDim.x >= 128);
	if (tid < 128) {
		negCenter(tid) = -float(acc[tid]) / nbDesc;
	}
	__syncthreads();
	static constexpr uint32_t grpSize = 8;
	assert(blockDim.x == grpSize * 64);
	const auto g = cg::tiled_partition<grpSize>(cg::this_thread_block());
	const uint32_t idxGrp = tid / grpSize;
	const uint32_t idxLane = tid % grpSize; assert(idxLane == g.thread_rank());
	const cudapp::KArray<int8_t, 16> w = reinterpret_cast<const cudapp::KArray<int8_t, grpSize, 16>&>(weights[idxGrp])[idxLane];
	float v = 0;
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		const cudapp::KArray<float, 4> c = reinterpret_cast<const cudapp::KArray<float, 4>&>(negCenter(idxLane * 16 + i * 4));
		#pragma unroll
		for (int j = 0; j < 4; j++) {
			v += c[j] * w[i * 4 + j];
		}
	}
	#pragma unroll
	for (uint32_t laneMask = grpSize / 2; laneMask != 0; laneMask /= 2) {
		v += g.shfl_xor(v, laneMask);
	}
	if (idxLane == 0) {
		bias[idxGrp] = int32_t(round(v)) + biasBase[idxGrp];
	}
}

void computeBias(cudapp::KArray<int32_t, 64>& __restrict__ bias,
	const cudapp::KArray<int32_t, 64, 32>& __restrict__ weights, const cudapp::KArray<int32_t, 64>& __restrict__ biasBase,
	const cudapp::KArray<uint64_t, 128>& __restrict__ acc, uint64_t nbDesc, cudaStream_t stream)
{
	kernelComputeBias<<<1, ctaSize_computeBias, 0, stream>>>(bias, weights, biasBase, acc, nbDesc);
	cudaCheck(cudaGetLastError());
}

} // rsfm::cashash
