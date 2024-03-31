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

void testCasHashFindBestMatch(uint32_t nbDesc, bool refCheck, uint32_t nbQueries){
	const uint32_t nbWarmUpRuns = refCheck ? 0 : 1;
	const uint32_t repeat = refCheck ? 1 : 16;

	const uint32_t totalNbDesc = nbDesc * (nbQueries + 1);

	using Descriptor = KArray<uint32_t, 32>;

	CudaMem<Descriptor, CudaMemType::kManaged> descriptors = allocCudaMem<Descriptor, CudaMemType::kManaged>(totalNbDesc);
	std::uniform_int_distribution<uint32_t> distDescWord(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
	std::generate_n(descriptors.get(), totalNbDesc, [&](){
		Descriptor desc;
		for (int i = 0; i < 32; i++) {
			desc[i] = distDescWord(global::random_engine);
		}
		return desc;
	});
	CudaMem<uint32_t, CudaMemType::kManaged> sqrNormDesc = allocCudaMem<uint32_t, CudaMemType::kManaged>(totalNbDesc);

	CudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged> bucketIndices = allocCudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged>(totalNbDesc);
	std::uniform_int_distribution<uint32_t> distBucketIndices{0, 255};
	std::generate_n(bucketIndices.get(), totalNbDesc, [&](){
		KArray<uint8_t, 8> ret;
		for (int i = 0; i < 8; i++)
			ret[i] = distBucketIndices(global::random_engine);
		return ret;
	});
	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> bucketEnds = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(nbQueries + 1);
	CudaMem<uint32_t, CudaMemType::kManaged> buckets = allocCudaMem<uint32_t, CudaMemType::kManaged>(totalNbDesc * 8);
	const auto maxBucketSizes = allocCudaMem<uint32_t, CudaMemType::kManaged>(nbQueries + 1);

	const cudaStream_t stream = nullptr;
    const int device = getCudaDevice();

	cudaCheck(cudaMemPrefetchAsync(sqrNormDesc.get(), sizeof(sqrNormDesc[0]) * totalNbDesc, device, stream));
	launchPreCompSiftSqrNorm(sqrNormDesc.get(), descriptors.get(), totalNbDesc, stream);

	const auto resultBuf = allocCudaMem<cashash::BestMatch, CudaMemType::kManaged>(nbDesc * nbQueries * 2);
	cudaCheck(cudaMemPrefetchAsync(resultBuf.get(), sizeof(resultBuf[0]) * nbDesc * nbQueries * 2, device, stream));

	cudaCheck(cudaMemPrefetchAsync(bucketIndices.get(), sizeof(bucketIndices[0]) * totalNbDesc, device, stream));
	cudaCheck(cudaMemPrefetchAsync(bucketEnds.get(), sizeof(bucketEnds[0]) * (nbQueries + 1), device, stream));
	cudaCheck(cudaMemPrefetchAsync(buckets.get(), sizeof(buckets[0]) * totalNbDesc * 8, device, stream));
	cudaCheck(cudaMemPrefetchAsync(maxBucketSizes.get(), sizeof(maxBucketSizes[0]) * (nbQueries + 1), device, stream));

	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> histogram = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(1);
	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> bucketBeg = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(1);
	cudaCheck(cudaMemPrefetchAsync(histogram.get(), sizeof(histogram[0]), device, stream));
	cudaCheck(cudaMemPrefetchAsync(bucketBeg.get(), sizeof(bucketBeg[0]), device, stream));
	for (uint32_t i = 0; i < nbQueries + 1; i++) {
		cashash::buildHistogram(histogram.get(), &bucketIndices[nbDesc * i], nbDesc, stream);
		cashash::computeBucketBound(bucketBeg[0], bucketEnds[i], &maxBucketSizes[i], histogram[0], stream);
		cashash::buildBuckets(&bucketIndices[nbDesc * i], nbDesc, bucketBeg[0], &buckets[nbDesc * i], stream);
	}
	cudaCheck(cudaMemPrefetchAsync(maxBucketSizes.get(), sizeof(maxBucketSizes[0]) * (nbQueries + 1), cudaCpuDeviceId, stream));
	cudaCheck(cudaStreamSynchronize(stream)); // need maxBucketSizes on host.

	std::vector<cashash::ImageDesc> images(nbQueries + 1);
	std::vector<uint32_t> imgNbDesc(nbQueries + 1);
	std::vector<cashash::MatchResult> results(nbQueries);
	for (uint32_t i = 0; i < nbQueries + 1; i++) {
		images.at(i) = cashash::ImageDesc{
			&descriptors[nbDesc * i],
			&bucketEnds[i],
			&buckets[nbDesc * i],
			&sqrNormDesc[nbDesc * i]
		};
		imgNbDesc.at(i) = nbDesc;
		if (i < nbQueries) {
			results.at(i) = cashash::MatchResult{
				&resultBuf[nbDesc * 2 * i],
				&resultBuf[nbDesc * (2 * i + 1)]
			};
		}
	}

	auto run = [&](){
		cashash::findBestMatch(&images[0], &imgNbDesc[0], images.back(), imgNbDesc.back(), &results[0], nbQueries, &maxBucketSizes[0], maxBucketSizes[nbQueries], stream);
	};

    const auto event_start = makeCudaEvent(cudaEventDefault);
    const auto event_end = makeCudaEvent(cudaEventDefault);

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
		printf("performance: %f FPS\n", repeat * nbQueries / time * 1000.f);
	}

    if (refCheck) {
        printf("Checking results...\n");
		auto getL2Sqr = [](const Descriptor& a, const Descriptor& b){
			uint32_t result = 0;
			for(int i = 0; i < (int)sizeof(Descriptor); i++){
				result += square(int(reinterpret_cast<const uint8_t*>(&a)[i]) - int(reinterpret_cast<const uint8_t*>(&b)[i]));
			}
			return result;
		};
		for (uint32_t idxQuery = 0; idxQuery < nbQueries; idxQuery++) {
			const auto& query = images.at(idxQuery);
			const auto& train = images.back();
			std::vector<cashash::BestMatch> qMatches(nbDesc);
			std::vector<cashash::BestMatch> tMatches(nbDesc);
			std::fill_n(qMatches.begin(), nbDesc, cashash::BestMatch{~0u, ~0u});
			std::fill_n(tMatches.begin(), nbDesc, cashash::BestMatch{~0u, ~0u});
			for(uint32_t idxBucket = 0; idxBucket < 256; idxBucket++) {
				for (uint32_t n = 0; n < 8; n++) {
					const uint32_t qBucketSize = query.bucketSize(n, idxBucket);
					const uint32_t tBucketSize = train.bucketSize(n, idxBucket);
					const uint32_t* qBucket = &query.buckets[query.bucketBeg(n, idxBucket)];
					const uint32_t* tBucket = &train.buckets[train.bucketBeg(n, idxBucket)];

					for (uint32_t i = 0; i < qBucketSize; i++) {
						const uint32_t idxQDesc = qBucket[i];
						const Descriptor& qDesc = query.desc[idxQDesc];
						for (uint32_t j = 0; j < tBucketSize; j++) {
							const uint32_t idxTDesc = tBucket[j];
							const Descriptor& tDesc = train.desc[idxTDesc];
							const uint32_t distance = getL2Sqr(qDesc, tDesc);
							// printf("%u: %u - %u\n", distance, idxQDesc, idxTDesc);
							if (distance < qMatches.at(idxQDesc).distance) {
								qMatches.at(idxQDesc) = {idxTDesc, distance};
							}
							if (distance < tMatches.at(idxTDesc).distance) {
								tMatches.at(idxTDesc) = {idxQDesc, distance};
							}
						}
					}
				}
			}
			const auto& res = results.at(idxQuery);
			for (uint32_t i = 0; i < nbDesc; i++) {
				EXPECT_EQ(qMatches.at(i).index, res.query[i].index);
				EXPECT_EQ(qMatches.at(i).distance, res.query[i].distance);
				EXPECT_EQ(tMatches.at(i).index, res.train[i].index);
				EXPECT_EQ(tMatches.at(i).distance, res.train[i].distance);
			}
		}
    }
}

TEST(testCasHashFindBestMatch, RefCheck)
{
	// testCasHashFindBestMatch(16, true, 1);
	testCasHashFindBestMatch(897, true, 2);
	testCasHashFindBestMatch(164, true, 40);
}

TEST(testCasHashFindBestMatch, Perf)
{
	testCasHashFindBestMatch(20000, false, 48);
}
