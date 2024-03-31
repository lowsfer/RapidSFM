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

#include <gtest/gtest.h>
#include "../bruteForceMatch.h"
#include <random>
#include <cassert>
#include <cuda_utils.h>
#include <macros.h>
#include "../casHash/cascadeHash.h"

using namespace rsfm;
using namespace cudapp;

namespace global {
extern unsigned random_seed;
extern std::default_random_engine random_engine;
}

void testComputeCasHash(size_t nbDesc, bool refCheck){
	const size_t nbWarmUpRuns = refCheck ? 0 : 32;
	const size_t repeat = refCheck ? 1 : 128;

	CudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged> bucketIndices = allocCudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged>(nbDesc);
	std::uniform_int_distribution<uint32_t> distBucketIndices{0, 255};
	std::generate_n(bucketIndices.get(), nbDesc, [&](){
		KArray<uint8_t, 8> ret;
		for (int i = 0; i < 8; i++)
			ret[i] = distBucketIndices(global::random_engine);
		return ret;
	});
	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> histogram = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(1);
	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> bucketBounds = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(2);
	CudaMem<uint32_t, CudaMemType::kManaged> buckets = allocCudaMem<uint32_t, CudaMemType::kManaged>(nbDesc * 8);
	CudaMem<uint32_t, CudaMemType::kPinned> maxBucketSize = allocCudaMem<uint32_t, CudaMemType::kPinned>(1);

    const cudaStream_t stream = 0;
    const auto event_start = makeCudaEvent(cudaEventDefault);
    const auto event_end = makeCudaEvent(cudaEventDefault);
    const int device = getCudaDevice();

	cudaCheck(cudaMemPrefetchAsync(bucketIndices.get(), sizeof(bucketIndices[0]) * nbDesc, device, stream));
	cudaCheck(cudaMemPrefetchAsync(histogram.get(), sizeof(histogram[0]), device, stream));
	cudaCheck(cudaMemPrefetchAsync(bucketBounds.get(), sizeof(bucketBounds[0]) * 2, device, stream));
	// cudaCheck(cudaMemPrefetchAsync(maxBucketSize.get(), sizeof(maxBucketSize[0]), device, stream));

	auto run = [&](){
		cashash::buildHistogram(histogram.get(), bucketIndices.get(), nbDesc, stream);
		cashash::computeBucketBound(bucketBounds[0], bucketBounds[1], maxBucketSize.get(), histogram[0], stream);
		cashash::buildBuckets(bucketIndices.get(), nbDesc, bucketBounds[0], buckets.get(), stream);
	};

    // warm up
    for(size_t n = 0; n < nbWarmUpRuns; n++) {
		run();
    }
    cudaCheck(cudaEventRecord(event_start.get(), stream));
    for(size_t n = 0; n < repeat; n++) {
        run();
    }
    cudaCheck(cudaEventRecord(event_end.get(), stream));
    cudaCheck(cudaStreamSynchronize(stream));
	if (!refCheck)
	{
		cudaCheck(cudaEventSynchronize(event_start.get()));
		cudaCheck(cudaEventSynchronize(event_end.get()));
		float time;
		cudaCheck(cudaEventElapsedTime(&time, event_start.get(), event_end.get()));
		printf("performance: %f FPS\n", repeat / time * 1000.f);
	}

    if (refCheck) {
        printf("Checking results...\n");
		uint32_t refMax = 0u;
		for (uint32_t n = 0; n < 8; n++) {
			for (uint32_t i = 0; i < 256; i++) {
				EXPECT_EQ(bucketBounds[0][n][i], bucketBounds[1][n][i]);
				const uint32_t bucketSize = bucketBounds[1][n][i] - (i == 0 ? 0 : bucketBounds[1][n][i-1]);
				refMax = std::max(bucketSize, refMax);
			}
		}
		EXPECT_EQ(refMax, maxBucketSize[0]);
		for (uint32_t n = 0; n < 8; n++) {
			uint32_t idxBucket = 0;
			for(uint32_t i = 0; i < nbDesc; i++) {
				while (i >= bucketBounds[1][n][idxBucket]) {
					idxBucket++;
				}
				const auto idxDesc = buckets[nbDesc * n + i];
				EXPECT_EQ(bucketIndices[idxDesc][n], idxBucket);
			}
		}
    }
}

TEST(testCasHashBuildBuckets, RefCheck)
{
	testComputeCasHash(897, true);
	testComputeCasHash(164, true);
}
TEST(testCasHashBuildBuckets, Perf)
{
	testComputeCasHash(30000, false);
}
