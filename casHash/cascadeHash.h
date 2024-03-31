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
#include <KArray.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "../bruteForceMatch.h"

namespace rsfm::cashash
{

static constexpr uint32_t nbScheme = 8;
static constexpr uint32_t nbBuckets = 256;

// @fixme: should accumulate for covariance, not just variance! so sqrAcc is currently unused for now.
// every desc element is divided by 2, like what we do in computeHash.
void accumuateDesc(cudapp::KArray<uint64_t, 128>& acc, cudapp::KArray<uint64_t, 128>* __restrict__ sqrAcc, bool initAcc, const cudapp::KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc, cudaStream_t stream);
void computeBias(cudapp::KArray<int32_t, 64>& __restrict__ bias,
	const cudapp::KArray<int32_t, 64, 32>& __restrict__ weights, const cudapp::KArray<int32_t, 64>& __restrict__ biasBase,
	const cudapp::KArray<uint64_t, 128>& __restrict__ acc, uint64_t nbDesc, cudaStream_t stream);

// 64 channels
// @fixme: not implemented
void createWeightsBiasForBucketOnly(cudapp::KArray<int32_t, 32>* __restrict__ weights, int32_t* __restrict__ bias,
	const float* __restrict__ randVal, // length = 128 * 64 + 64, normal distribution N(0, 1)
	const cudapp::KArray<uint64_t, 128>& acc, uint32_t nbAccDesc, cudaStream_t stream);

// For better performance, set histogram to nullptr and use buildHistogram.
// 192 channels
void computeHashForBucketAndRemap(const cudapp::KArray<int32_t, 32>* __restrict__ weights, const int32_t* __restrict__ bias,
	const cudapp::KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc,
    cudapp::KArray<uint8_t, nbScheme>* __restrict__ bucketIndices, cudapp::KArray<uint32_t, 4>* __restrict__ remap,
	cudapp::KArray<uint32_t, nbScheme, nbBuckets>* __restrict__ histogram, cudaStream_t stream);

// For better performance, set histogram to nullptr and use buildHistogram.
// 64 channels
void computeHashForBucketOnly(const cudapp::KArray<int32_t, 32>* __restrict__ weights, const int32_t* __restrict__ bias,
	const cudapp::KArray<uint32_t, 32>* __restrict__ desc, uint32_t nbDesc,
    cudapp::KArray<uint8_t, nbScheme>* __restrict__ bucketIndices,
	cudapp::KArray<uint32_t, nbScheme, nbBuckets>* __restrict__ histogram, cudaStream_t stream);

// Faster than fused histogram computation in hash computation
void buildHistogram(cudapp::KArray<uint32_t, nbScheme, nbBuckets>* __restrict__ histogram,
    const cudapp::KArray<uint8_t, nbScheme>* __restrict__ bucketIndices, uint32_t nbDesc, cudaStream_t stream);

void computeBucketBound(cudapp::KArray<uint32_t, nbScheme, nbBuckets>& __restrict__ lower, cudapp::KArray<uint32_t, nbScheme, nbBuckets>& __restrict__ upper,
	uint32_t* __restrict__ maxBucketSize, const cudapp::KArray<uint32_t, nbScheme, nbBuckets>& __restrict__ histogram, cudaStream_t stream);

void buildBuckets(const cudapp::KArray<uint8_t, nbScheme>* __restrict__ bucketIndices, uint32_t nbDesc,
	cudapp::KArray<uint32_t, nbScheme, nbBuckets>& __restrict__ idxAllocator,
	uint32_t* __restrict__ buckets, // shape is [nbScheme][nbDesc]
	cudaStream_t stream);

struct ImageDesc
{
	static constexpr DescElemType descElemType = DescElemType::kU8;
	static constexpr size_t descDims = 128;
	static constexpr DistanceStyle distStyle = DistanceStyle::kL2;
    using Descriptor = typename DescTraits<descElemType, descDims, distStyle>::Descriptor;
    using Distance = typename DescTraits<descElemType, descDims, distStyle>::Distance;

	__device__ __host__ inline
    uint32_t bucketSize(uint32_t idxScheme, uint32_t idxBucket) const {
		return bucketEnds[0][idxScheme][idxBucket] - bucketBeg(idxScheme, idxBucket);
	};
	__device__ __host__ inline
	uint32_t bucketBeg(uint32_t idxScheme, uint32_t idxBucket) const {
		return idxBucket == 0 ? 0u : bucketEnds[0][idxScheme][idxBucket - 1];
	}
	__device__ __host__ inline
	uint32_t nbDesc() const {
		return bucketEnds[0][0][255];
	}

    const Descriptor* __restrict__ desc;

	const cudapp::KArray<uint32_t, nbScheme, nbBuckets>* __restrict__ bucketEnds;
	const uint32_t* __restrict__ buckets; // shape is [nbScheme][nbDesc]

    // Used when we use dot product to implement L2. Square norm of each desc can be pre-computed.
    const uint32_t* __restrict__ sqrNormDesc;
};

using BestMatch = BestMatchImpl<uint32_t>;
struct MatchResult
{
	BestMatch* __restrict__ query; // indices are to train
	BestMatch* __restrict__ train; // indices are to query
};

void findBestMatch(const ImageDesc* queries, const uint32_t* querySize, const ImageDesc& train, uint32_t trainSize, const MatchResult* results, uint32_t nbTasks,
	const uint32_t* maxQueryBucketSize, uint32_t maxTrainBucketSize, cudaStream_t stream);

}