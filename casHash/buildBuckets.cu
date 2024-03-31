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
#include <KArray.h>
#include <cuda_utils.h>

namespace rsfm::cashash
{
using namespace cudapp;

constexpr uint32_t ctaSize_buildHisto = 256;
constexpr uint32_t unroll_buildHisto = 8;
__global__ void kernelBuildHistogram(cudapp::KArray<uint32_t, 8, 256>* __restrict__ histogram,
    const cudapp::KArray<uint8_t, 8>* __restrict__ bucketIndices, uint32_t nbDesc)
{
	assert(ctaSize_buildHisto == blockDim.x);
	const auto idxDescBase = ctaSize_buildHisto * unroll_buildHisto * blockIdx.x;
	if (idxDescBase >= nbDesc) {
		return;
	}
#if 0
	// this is slowest option
	#pragma unroll
	for (uint32_t i = 0; i < unroll_buildHisto; i++) {
		const auto idxDesc = idxDescBase + ctaSize_buildHisto * i + threadIdx.x;
		const bool isInRange = idxDesc < nbDesc;
		if (!isInRange) {
			break;
		}
		const cudapp::KArray<uint8_t, 8> indices = bucketIndices[idxDesc];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			atomicAdd(&histogram[0][i][indices[i]], 1);
		}
	}
#else
	__shared__ KArray<uint32_t, 8, 256> smemHist;
	constexpr uint32_t histWords = sizeof(smemHist) / sizeof(smemHist[0][0]);
	#pragma unroll
	for (uint32_t i = 0; i < divUp(histWords, ctaSize_buildHisto); i++) {
		const auto idx = ctaSize_buildHisto * i + threadIdx.x;
		if (histWords % ctaSize_buildHisto != 0 && idx >= histWords) {
			break;
		}
		(&smemHist[0][0])[idx] = 0u;
	}
	__syncthreads();
	#pragma unroll
	for (uint32_t i = 0; i < unroll_buildHisto; i++) {
		const auto idxDesc = idxDescBase + ctaSize_buildHisto * i + threadIdx.x;
		const bool isInRange = idxDesc < nbDesc;
#if 1
		if (!isInRange) {
			break;
		}
		const cudapp::KArray<uint8_t, 8> indices = bucketIndices[idxDesc];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			atomicAdd_block(&smemHist[i][indices[i]], 1);
		}
#else
		// this is significantly slower
		const cudapp::KArray<uint8_t, 8> indices = isInRange ? bucketIndices[idxDesc] : cudapp::KArray<uint8_t, 8>{};
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			const auto idx = isInRange ? indices[i] : ~0U;
			const uint32_t match = __match_any_sync(~0U, idx);
			assert(match != 0);
			if (isInRange) {
				const uint32_t nbMatch = __popc(match);
				const bool isFirstBit = (lane_id() + 1 == __ffs(match));
				if (isFirstBit) {
					atomicAdd_block(&smemHist[i][idx], nbMatch);
				}
			}
		}
#endif
	}
	__syncthreads();
	#pragma unroll
	for (uint32_t i = 0; i < divUp(histWords, ctaSize_buildHisto); i++) {
		const auto idx = ctaSize_buildHisto * i + threadIdx.x;
		if (histWords % ctaSize_buildHisto != 0 && idx >= histWords) {
			break;
		}
		atomicAdd(&(&histogram[0][0][0])[idx], (&smemHist[0][0])[idx]);
	}
#endif
}

void buildHistogram(cudapp::KArray<uint32_t, 8, 256>* __restrict__ histogram,
    const cudapp::KArray<uint8_t, 8>* __restrict__ bucketIndices, uint32_t nbDesc, cudaStream_t stream)
{
	cudaCheck(cudaMemsetAsync(histogram, 0, sizeof(*histogram), stream));
	// launchKernel(&kernelBuildHistogram, divUp(nbDesc, ctaSize_buildHisto * unroll_buildHisto), ctaSize_buildHisto, size_t{0}, stream, histogram, bucketIndices, nbDesc);
	kernelBuildHistogram<<<divUp(nbDesc, ctaSize_buildHisto * unroll_buildHisto), ctaSize_buildHisto, size_t{0}, stream>>>(histogram, bucketIndices, nbDesc);
	cudaCheck(cudaGetLastError());
}

__launch_bounds__(256, 1)
__global__ void kernelComputeBucketBounds(KArray<uint32_t, 8, 256>& __restrict__ lower, KArray<uint32_t, 8, 256>& __restrict__ upper, uint32_t* __restrict__ maxBucketSize, const KArray<uint32_t, 8, 256>& __restrict__ histogram) {
	assert(blockDim.x == 256);

	__shared__ KArray<uint32_t, 8, 256> bufData;
	auto buf = [&](uint32_t i, uint32_t j) -> uint32_t& {
		return bufData[i][(j + i) % 256];
	};
	const uint32_t tid = threadIdx.x;
	#pragma unroll
	for (uint32_t n = 0; n < 8; n++) {
		buf(n, tid) = histogram[n][tid];
	}
	__syncthreads();
	if (tid < 8) {
		uint32_t sum = 0;
		uint32_t thrdMaxBucketSize = 0u;
		#pragma unroll
		for (int i = 0; i < 256; i++) {
#if 0
			const uint32_t bucketSize = buf(tid, i);
			sum += bucketSize;
			buf(tid, i) = sum;
#else
			const uint32_t bucketSize = atomicAdd(&buf(tid, i), sum);
			sum += bucketSize;
#endif
			if (bucketSize > thrdMaxBucketSize) {
				thrdMaxBucketSize = bucketSize;
			}
		}
#if __CUDA_ARCH__ >= 800
		const uint32_t glbMaxBucketSize = __reduce_max_sync(0xFFU, thrdMaxBucketSize);
#else
		uint32_t glbMaxBucketSize = thrdMaxBucketSize;
		#pragma unroll
		for (uint32_t xorMask = 4; xorMask != 0; xorMask /= 2) {
			const uint32_t other = __shfl_xor_sync(0xFFU, glbMaxBucketSize, xorMask);
			glbMaxBucketSize = std::max(glbMaxBucketSize, other);
		}
#endif
		if (tid == 0) {
			*maxBucketSize = glbMaxBucketSize;
		}
	}
	__syncthreads();
	#pragma unroll
	for (uint32_t n = 0; n < 8; n++) {
		lower[n][tid] = tid == 0 ? 0u : buf(n, tid - 1);
		upper[n][tid] = buf(n, tid);
	}
}

void computeBucketBound(KArray<uint32_t, 8, 256>& __restrict__ lower, KArray<uint32_t, 8, 256>& __restrict__ upper, uint32_t* __restrict__ maxBucketSize, const KArray<uint32_t, 8, 256>& __restrict__ histogram, cudaStream_t stream) {
	kernelComputeBucketBounds<<<1, 256, 0, stream>>>(lower, upper, maxBucketSize, histogram);
	cudaCheck(cudaGetLastError());
}

using SiftDesc = KArray<uint32_t, 32>;

constexpr uint32_t ctaSize_buildBuckets = 128;
constexpr uint32_t unroll_buildBuckets = 4;
__global__ void kernelBuildBuckets(const KArray<uint8_t, 8>* __restrict__ bucketIndices, uint32_t nbDesc,
	KArray<uint32_t, 8, 256>& __restrict__ idxAllocator,
	uint32_t* __restrict__ buckets // shape is [8][nbDesc]
	)
{
	assert(ctaSize_buildBuckets == blockDim.x);
	const auto idxDescBase = ctaSize_buildBuckets * unroll_buildBuckets * blockIdx.x;
#if 0
	// native implementation
	if (idxDescBase >= nbDesc) {
		return;
	}
	#pragma unroll
	for (uint32_t i = 0; i < unroll_buildBuckets; i++) {
		const auto idxDesc = idxDescBase + ctaSize_buildBuckets * i + threadIdx.x;
		const bool isInRange = idxDesc < nbDesc;
		if (!isInRange) {
			break;
		}
		const cudapp::KArray<uint8_t, 8> indices = bucketIndices[idxDesc];
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			const uint32_t idxBucket = indices[i];
			const auto idxSlot = atomicAdd(&idxAllocator[i][idxBucket], 1);
			buckets[nbDesc * i + idxSlot] = idxDesc;
		}
	}
#else
	// two pass: 1. build cta histogram and create cta index allocator; 2. fill buckets

	__shared__ KArray<uint32_t, 8, 256> ctaHist;
	const uint32_t tid = threadIdx.x;
	static_assert(sizeof(ctaHist) / sizeof(uint4) % ctaSize_buildBuckets == 0);
	#pragma unroll
	for (uint32_t i = 0; i < sizeof(ctaHist) / sizeof(uint4) / ctaSize_buildBuckets; i++) {
		reinterpret_cast<uint4*>(&ctaHist)[ctaSize_buildBuckets * i + tid] = uint4{0, 0, 0, 0};
	}
	__syncthreads();
	KArray<uint64_t, unroll_buildBuckets> thrdCache; // avoid duplicate load of two passes.
	// same implementation as kernelBuildHistogram
	#pragma unroll
	for (uint32_t i = 0; i < unroll_buildBuckets; i++) {
		const auto idxDesc = idxDescBase + ctaSize_buildBuckets * i + tid;
		const bool isInRange = idxDesc < nbDesc;
		if (!isInRange) {
			break;
		}
		thrdCache[i] = reinterpret_cast<const uint64_t&>(bucketIndices[idxDesc]);
		const cudapp::KArray<uint8_t, 8> indices = reinterpret_cast<const cudapp::KArray<uint8_t, 8>&>(thrdCache[i]);
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			atomicAdd_block(&ctaHist[i][indices[i]], 1);
		}
	}
	__syncthreads();
	KArray<uint32_t, 8, 256>& ctaIdxAllocator = ctaHist;
	static_assert(sizeof(ctaIdxAllocator) / sizeof(uint32_t) % ctaSize_buildBuckets == 0);
	#pragma unroll
	for (uint32_t i = 0; i < sizeof(ctaIdxAllocator) / sizeof(uint32_t) / ctaSize_buildBuckets; i++) {
		const uint32_t idx = i * ctaSize_buildBuckets + tid;
		const uint32_t ctaBucketSize = (&ctaHist[0][0])[idx];
		const uint32_t ctaBucketIdxBeg = atomicAdd(&(&idxAllocator[0][0])[idx], ctaBucketSize);
		(&ctaIdxAllocator[0][0])[idx] = ctaBucketIdxBeg;
	}
	__syncthreads();
	// the second pass
	#pragma unroll
	for (uint32_t i = 0; i < unroll_buildBuckets; i++) {
		const auto idxDesc = idxDescBase + ctaSize_buildBuckets * i + threadIdx.x;
		const bool isInRange = idxDesc < nbDesc;
		if (!isInRange) {
			break;
		}
		const cudapp::KArray<uint8_t, 8> indices = reinterpret_cast<const cudapp::KArray<uint8_t, 8>&>(thrdCache[i]);
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			const uint32_t idxBucket = indices[i];
			const auto idxSlot = atomicAdd_block(&ctaIdxAllocator[i][idxBucket], 1);
			buckets[nbDesc * i + idxSlot] = idxDesc;
		}
	}
#endif
}

void buildBuckets(const KArray<uint8_t, 8>* __restrict__ bucketIndices, uint32_t nbDesc,
	KArray<uint32_t, 8, 256>& __restrict__ idxAllocator,
	uint32_t* __restrict__ buckets, // shape is [8][nbDesc]
	cudaStream_t stream)
{
	kernelBuildBuckets<<<divUp(nbDesc, ctaSize_buildBuckets * unroll_buildBuckets), ctaSize_buildBuckets, 0, stream>>>(bucketIndices, nbDesc, idxAllocator, buckets);
	cudaCheck(cudaGetLastError());
}

}