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
// Created by yao on 28/01/18.
//
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


template <bool makeRemap>
void testComputeCasHash(size_t nbDesc, bool refCheck, bool fuseHisto){
	const size_t nbQueryDesc = nbDesc;

	const size_t nbWarmUpRuns = refCheck ? 0 : 32;
	const size_t repeat = refCheck ? 1 : 128;
	constexpr size_t nbChannels = makeRemap ? 192 : 64;

	static_assert(nbChannels == 64 || nbChannels == 192);
	using Channel = KArray<int32_t, 32>;
    using Descriptor = KArray<uint32_t, 32>;
    const CudaMem<Channel, CudaMemType::kManaged> weight = allocCudaMem<Channel, CudaMemType::kManaged, false>(nbChannels);
    CudaMem<int32_t, CudaMemType::kManaged> bias = allocCudaMem<int32_t, CudaMemType::kManaged, false>(nbChannels);
    std::uniform_int_distribution<uint32_t> dist_desc_word(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    auto gen_desc = [&](){
        Descriptor result;
        for(auto& w : result)
            w = dist_desc_word(global::random_engine);
        return result;
    };
    std::generate_n(weight.get(), nbChannels, [&gen_desc](){
		auto desc = gen_desc();
		Channel w;
		memcpy(&w, &desc, sizeof(w));
		return w;
	});
	for (uint32_t i = 0; i < nbChannels; i++) {
		const auto& w = reinterpret_cast<const KArray<int8_t, 128>&>(weight[i]);
		bias[i] = static_cast<int32_t>(roundf(std::accumulate(w.begin(), w.end(), 0, [](float acc, int8_t x){return acc + x * -63.5f;})));
	}
    CudaMem<Descriptor, CudaMemType::kManaged> query = allocCudaMem<Descriptor, CudaMemType::kManaged, false>(nbQueryDesc);
	std::generate_n(query.get(), nbQueryDesc, gen_desc);
	CudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged> bucketIndices = allocCudaMem<KArray<uint8_t, 8>, CudaMemType::kManaged>(nbQueryDesc);
	CudaMem<KArray<uint32_t, 4>, CudaMemType::kManaged> remap = allocCudaMem<KArray<uint32_t, 4>, CudaMemType::kManaged>(nbQueryDesc);
	CudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged> histogram = allocCudaMem<KArray<uint32_t, 8, 256>, CudaMemType::kManaged>(1);

    const cudaStream_t stream = 0;
    const auto event_start = makeCudaEvent(cudaEventDefault);
    const auto event_end = makeCudaEvent(cudaEventDefault);
    const int device = getCudaDevice();

    cudaCheck(cudaMemPrefetchAsync(weight.get(), sizeof(Channel) * nbChannels, device, stream));
	cudaCheck(cudaMemPrefetchAsync(bias.get(), sizeof(uint32_t) * nbChannels, device, stream));
	cudaCheck(cudaMemPrefetchAsync(query.get(), sizeof(Descriptor) * nbQueryDesc, device, stream));
	cudaCheck(cudaMemPrefetchAsync(bucketIndices.get(), sizeof(bucketIndices[0]) * nbQueryDesc, device, stream));
	cudaCheck(cudaMemPrefetchAsync(remap.get(), sizeof(remap[0]) * nbQueryDesc, device, stream));
	cudaCheck(cudaMemPrefetchAsync(histogram.get(), sizeof(histogram[0]), device, stream));
	cudaCheck(cudaMemsetAsync(bucketIndices.get(), 0, sizeof(bucketIndices[0]) * nbDesc, stream));
	cudaCheck(cudaMemsetAsync(remap.get(), 0, sizeof(remap[0]) * nbDesc, stream));

	auto run = [&](){
		auto fusedHistoPtr = fuseHisto ? histogram.get() : nullptr;
		if constexpr (nbChannels == 192) {
        	cashash::computeHashForBucketAndRemap(weight.get(), bias.get(), query.get(), nbQueryDesc, bucketIndices.get(), remap.get(), fusedHistoPtr, stream);
		}
		else {
			cashash::computeHashForBucketOnly(weight.get(), bias.get(), query.get(), nbQueryDesc, bucketIndices.get(), fusedHistoPtr, stream);
		}
		if (!fuseHisto) {
			cashash::buildHistogram(histogram.get(), bucketIndices.get(), nbQueryDesc, stream);
		}
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

    auto getBit = [](const Channel& w, const int32_t b, const Descriptor& desc){
        int32_t acc = b;
        for(int i = 0; i < (int)sizeof(Descriptor); i++){
            acc += int(reinterpret_cast<const int8_t*>(&w)[i]) * (int(reinterpret_cast<const uint8_t*>(&desc)[i]) / 2);
        }
        return acc >= 0;
    };
    if (refCheck) {
        printf("Checking results...\n");
        for(uint32_t i = 0; i < nbQueryDesc; i++) {
			for (uint32_t j = 0; j < nbChannels; j++) {
				bool bit;
				if (j < 64) {
					bit = (bucketIndices[i][j / 8] >> (j % 8)) & 0x1UL;
				}
				else {
					const auto idxBit = j - 64;
					bit = (remap[i][idxBit / 32] >> (idxBit % 32)) & 0x1U;
				}
				const bool ref = getBit(weight[j], bias[j], query[i]);
				EXPECT_EQ(bit, ref);
			}
        }
		std::array<std::array<uint32_t, 256>, 8> refHist{};
		for (uint32_t i = 0; i < nbQueryDesc; i++) {
			for (int j = 0; j < 8; j++) {
				refHist[j][bucketIndices[i][j]]++;
			}
		}
		for (int j = 0; j < 8; j++) {
			for (uint32_t i = 0; i < 256; i++) {
				EXPECT_EQ(histogram[0][j][i], refHist[j][i]);
			}
		}
    }
}

TEST(testComputeCasHash, BucketRemapHistoRefCheck)
{
	testComputeCasHash<true>(897, true, true);
}

TEST(testComputeCasHash, BucketHistoRefCheck)
{
	testComputeCasHash<false>(897, true, true);
}
TEST(testComputeCasHash, BucketRemapRefCheck)
{
	testComputeCasHash<true>(897, true, false);
}

TEST(testComputeCasHash, BucketRefCheck)
{
	testComputeCasHash<false>(897, true, false);
}


TEST(testComputeCasHash, BucketRemapHistoPerf)
{
	testComputeCasHash<true>(30000, false, true);
}
TEST(testComputeCasHash, BucketRemapPerf)
{
	testComputeCasHash<true>(30000, false, false);
}

TEST(testComputeCasHash, BucketHistoPerf)
{
	testComputeCasHash<false>(30000, false, true);
}

TEST(testComputeCasHash, BucketPerf)
{
	testComputeCasHash<false>(30000, false, false);
}
