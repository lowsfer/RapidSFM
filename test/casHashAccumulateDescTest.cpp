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

void testCasHashAccumulateDesc(uint32_t nbDesc, bool refCheck){
	const uint32_t nbWarmUpRuns = refCheck ? 0 : 16;
	const uint32_t repeat = refCheck ? 1 : 128;

	using Descriptor = KArray<uint32_t, 32>;

	CudaMem<Descriptor, CudaMemType::kManaged> descriptors = allocCudaMem<Descriptor, CudaMemType::kManaged>(nbDesc);
	std::uniform_int_distribution<uint32_t> distDescWord(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
	std::generate_n(descriptors.get(), nbDesc, [&](){
		Descriptor desc;
		for (int i = 0; i < 32; i++) {
			desc[i] = distDescWord(global::random_engine);
		}
		return desc;
	});
	CudaMem<KArray<uint64_t, 128>, CudaMemType::kManaged> acc = allocCudaMem<KArray<uint64_t, 128>, CudaMemType::kManaged>(1);
	CudaMem<KArray<uint64_t, 128>, CudaMemType::kManaged> sqrAcc = allocCudaMem<KArray<uint64_t, 128>, CudaMemType::kManaged>(1);

	const cudaStream_t stream = nullptr;
    const int device = getCudaDevice();
	cudaCheck(cudaMemPrefetchAsync(descriptors.get(), sizeof(Descriptor) * nbDesc, device, stream));

	auto run = [&]() {
		cashash::accumuateDesc(acc[0], sqrAcc.get(), true, descriptors.get(), nbDesc, stream);
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
		printf("Performance: %f FPS\n", repeat / time * 1000.f);
	}

    if (refCheck) {
        printf("Checking results...\n");
		KArray<uint64_t, 128> refAcc{};
		KArray<uint64_t, 128> refSqrAcc{};
		for (uint32_t i = 0; i < nbDesc; i++) {
			const auto& desc = reinterpret_cast<const KArray<uint8_t, 128>&>(descriptors[i]);
			for (int i = 0; i < 128; i++) {
				const uint32_t val = desc[i] / 2u;
				refAcc[i] += val;
				refSqrAcc[i] += val * val;
			}
		}
		for (int i = 0; i < 128; i++) {
			EXPECT_EQ(refAcc[i], acc[0][i]);
			EXPECT_EQ(refSqrAcc[i], sqrAcc[0][i]);
		}
    }
}

TEST(testCasHashAccumulateDesc, RefCheck)
{
	testCasHashAccumulateDesc(2000, true);
}

TEST(testCasHashAccumulateDesc, Perf)
{
	testCasHashAccumulateDesc(30000, false);
}
/*
cudapp::KArray<int32_t, 64>& __restrict__ bias,
	const cudapp::KArray<int32_t, 64, 32>& __restrict__ weights, const cudapp::KArray<int32_t, 64>& __restrict__ biasBase,
	const cudapp::KArray<uint64_t, 128>& __restrict__ acc, uint32_t nbDesc, cudaStream_t stream
*/

void testComputeBias(uint32_t nbDesc, bool refCheck){
	const uint32_t nbWarmUpRuns = refCheck ? 0 : 16;
	const uint32_t repeat = refCheck ? 1 : 256;

	const auto bias = allocCudaMem<cudapp::KArray<int32_t, 64>, CudaMemType::kManaged>(1);
	const auto weights = allocCudaMem<cudapp::KArray<int32_t, 64, 32>, CudaMemType::kManaged>(1);
	{
		std::normal_distribution<float> dist(0, 32);
		for (unsigned i = 0; i < sizeof(weights[0]); i++) {
			reinterpret_cast<int8_t*>(weights.get())[i] = int8_t(clamp(dist(global::random_engine), -128.f, 127.1f));
		}
	}
	const auto biasBase = allocCudaMem<cudapp::KArray<int32_t, 64>, CudaMemType::kManaged>(1);
	{
		std::normal_distribution<float> dist(0, 16);
		for (int i = 0; i < 64; i++) {
			biasBase[0][i] = int32_t(clamp(dist(global::random_engine), (float)std::numeric_limits<int32_t>::lowest(), (float)std::numeric_limits<int32_t>::max()));
		}
	}
	const auto acc = allocCudaMem<cudapp::KArray<uint64_t, 128>, CudaMemType::kManaged>(1);
	{
		std::normal_distribution<float> dist(0, 16 * sqrt(float(nbDesc)));
		for (int i = 0; i < 64; i++) {
			biasBase[0][i] = uint64_t(clamp(dist(global::random_engine), (float)std::numeric_limits<uint64_t>::lowest(), (float)std::numeric_limits<uint64_t>::max()));
		}
	}

	const cudaStream_t stream = nullptr;
    const int device = getCudaDevice();
	cudaCheck(cudaMemPrefetchAsync(bias.get(), sizeof(bias[0]), device, stream));
	cudaCheck(cudaMemPrefetchAsync(weights.get(), sizeof(weights[0]), device, stream));
	cudaCheck(cudaMemPrefetchAsync(biasBase.get(), sizeof(biasBase[0]), device, stream));
	cudaCheck(cudaMemPrefetchAsync(acc.get(), sizeof(acc[0]), device, stream));

	auto run = [&]() {
		cashash::computeBias(bias[0], weights[0], biasBase[0], acc[0], nbDesc, stream);
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
		printf("Performance: %f FPS\n", repeat / time * 1000.f);
	}

    if (refCheck) {
        printf("Checking results...\n");
		KArray<float, 128> negCenter;
		for (int i = 0; i < 128; i++) {
			negCenter[i] = -float(acc[0][i]) / nbDesc;
		}
		for (uint32_t i = 0; i < 64; i++) {
			const auto& w = reinterpret_cast<const KArray<int8_t, 64, 128>&>(weights[0])[i];
			float v = 0.f;
			for (int j = 0; j < 128; j++) {
				v += w[j] * negCenter[j];
			}
			EXPECT_NEAR(int(round(v)) + biasBase[0][i], bias[0][i], 1.01);
		}
    }
}

TEST(testComputeBias, RefCheck)
{
	testComputeBias(2000, true);
}

TEST(testComputeBias, Perf)
{
	testComputeBias(30000, false);
}