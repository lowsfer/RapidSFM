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

using namespace rsfm;
using namespace cudapp;

namespace global {
//    unsigned random_seed = std::random_device{}();
unsigned random_seed = 0;
std::default_random_engine random_engine{random_seed};
}

// constexpr size_t nbDesc = 30720;
// constexpr size_t nbDesc = 897;
// constexpr size_t nbDesc = 12288;//1536;
// constexpr size_t nbDesc = 10000;
const int num_queries = 2;
const bool bidirectional_match = true;

void testBruteForceMatch(bool useIMMA, size_t nbDesc, bool refCheck){
	const size_t nbQueryDesc = nbDesc;
	const size_t nbTrainDesc = nbDesc;
	const size_t nbWarmUpRuns = refCheck ? 0 : 4;
	const size_t repeat = refCheck ? 1 : 128;
    using Descriptor = typename SiftDescTraits::Descriptor;
    using BestMatch = typename SiftBruteForceMatchTask::BestMatch;
    const CudaMem<Descriptor, CudaMemType::kManaged> train = allocCudaMem<Descriptor, CudaMemType::kManaged, false>(nbTrainDesc);
    CudaMem<uint32_t, CudaMemType::kManaged> trainSqrNorm;
    if (useIMMA) {
        trainSqrNorm = allocCudaMem<uint32_t, CudaMemType::kManaged, false>(nbTrainDesc);
    }
    std::uniform_int_distribution<uint32_t> dist_desc_word(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    auto gen_desc = [&](){
        Descriptor result;
        for(auto& w : result)
            w = dist_desc_word(global::random_engine);
        return result;
    };
    std::generate_n(train.get(), nbTrainDesc, gen_desc);
    CudaMem<Descriptor, CudaMemType::kManaged> queries[num_queries];
    CudaMem<uint32_t, CudaMemType::kManaged> queriesSqrNorm[num_queries];
    CudaMem<BestMatch, CudaMemType::kManaged> matches_fwd[num_queries];
    CudaMem<BestMatch, CudaMemType::kManaged> matches_bwd[num_queries];
    for(int i = 0; i < num_queries; i++){
        queries[i] = allocCudaMem<Descriptor, CudaMemType::kManaged, false>(nbQueryDesc);
        std::generate_n(queries[i].get(), nbQueryDesc, gen_desc);
        if (useIMMA) {
            queriesSqrNorm[i] = allocCudaMem<uint32_t, CudaMemType::kManaged, false>(nbQueryDesc);
        }
        matches_fwd[i] = allocCudaMem<BestMatch, CudaMemType::kManaged>(nbQueryDesc);
        if (bidirectional_match) {
            matches_bwd[i] = allocCudaMem<BestMatch, CudaMemType::kManaged>(nbTrainDesc);
        }
    }

    const cudaStream_t stream = 0;
    const auto event_start = makeCudaEvent(cudaEventDefault);
    const auto event_end = makeCudaEvent(cudaEventDefault);
    const int device = getCudaDevice();

    cudaCheck(cudaMemPrefetchAsync(train.get(), sizeof(Descriptor) * nbTrainDesc, device, stream));
    if (useIMMA) {
        cudaCheck(cudaMemPrefetchAsync(trainSqrNorm.get(), sizeof(uint32_t) * nbTrainDesc, device, stream));
        launchPreCompSiftSqrNorm(trainSqrNorm.get(), train.get(), nbTrainDesc, stream);
    }
    for(int i = 0; i < num_queries; i++){
        cudaCheck(cudaMemPrefetchAsync(queries[i].get(), sizeof(Descriptor) * nbQueryDesc, device, stream));
        if (useIMMA) {
            cudaCheck(cudaMemPrefetchAsync(queriesSqrNorm[i].get(), sizeof(uint32_t) * nbQueryDesc, device, stream));
            launchPreCompSiftSqrNorm(queriesSqrNorm[i].get(), queries[i].get(), nbQueryDesc, stream);
        }
        cudaCheck(cudaMemPrefetchAsync(matches_fwd[i].get(), sizeof(BestMatch) * nbQueryDesc, device, stream));
        cudaCheck(cudaMemPrefetchAsync(matches_bwd[i].get(), sizeof(BestMatch) * nbTrainDesc, device, stream));
    }

    std::vector<SiftBruteForceMatchTask> tasks(num_queries);
    for (int i = 0; i < num_queries; i++) {
        tasks[i] = SiftBruteForceMatchTask{
            cast32u(nbQueryDesc), cast32u(nbTrainDesc),
            queries[i].get(),
            train.get(),
            matches_fwd[i].get(),
            matches_bwd[i].get(),
            queriesSqrNorm[i].get(),
            trainSqrNorm.get()
        };
    }
    // warm up
    for(size_t n = 0; n < nbWarmUpRuns; n++) {
        if (useIMMA) {
            launchBruteForceMatchIMMA(tasks.data(), tasks.size(), stream);
        }
        else {
            launchBruteForceMatchIDP(tasks.data(), tasks.size(), stream);
        }
    }
    cudaCheck(cudaEventRecord(event_start.get(), stream));
    for(size_t n = 0; n < repeat; n++) {
        if (useIMMA) {
            launchBruteForceMatchIMMA(tasks.data(), tasks.size(), stream);
        }
        else {
            launchBruteForceMatchIDP(tasks.data(), tasks.size(), stream);
        }
    }
    cudaCheck(cudaEventRecord(event_end.get(), stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaEventSynchronize(event_start.get()));
    cudaCheck(cudaEventSynchronize(event_end.get()));
	if (!refCheck) {
		float time;
		cudaCheck(cudaEventElapsedTime(&time, event_start.get(), event_end.get()));
		printf("performance: %f FPS\n", repeat * num_queries / time * 1000.f);
	}

    auto getL2Sqr = [](const Descriptor& a, const Descriptor& b){
        uint32_t result = 0;
        for(int i = 0; i < (int)sizeof(Descriptor); i++){
            result += square(int(reinterpret_cast<const uint8_t*>(&a)[i]) - int(reinterpret_cast<const uint8_t*>(&b)[i]));
        }
        return result;
    };
    auto check_matches = [getL2Sqr](
            const Descriptor* query, size_t nbQueryDesc,
            const Descriptor* train, size_t nbTrainDesc,
            const BestMatch* matches){
        const size_t maxNbChecks = 1024;
        const size_t interval = std::max(1ul, nbQueryDesc / maxNbChecks);
        for(size_t i = 0; i < nbQueryDesc; i += interval){
            const auto& m = matches[i];
            ASSERT(m.index < nbTrainDesc);
            if (m.index < nbTrainDesc) {
                const auto refDistance = getL2Sqr(query[i], train[m.index]);
                HOPE(m.distance == refDistance);
                for (size_t j = 0; j < nbTrainDesc; j += interval) {
                    HOPE(getL2Sqr(query[i], train[j]) >= m.distance);
                }
            }
        }
    };
    if (refCheck) {
        for(int i = 0; i < num_queries; i++){
            check_matches(queries[i].get(), nbQueryDesc, train.get(), nbTrainDesc, matches_fwd[i].get());
            if(bidirectional_match)
                check_matches(train.get(), nbTrainDesc, queries[i].get(), nbQueryDesc, matches_bwd[i].get());
        }
    }
}

void testBruteForceMatchIMMA4b(size_t nbDesc, bool refCheck){
	const size_t nbQueryDesc = nbDesc;
	const size_t nbTrainDesc = nbDesc;
	const size_t nbWarmUpRuns = refCheck ? 0 : 4;
	const size_t repeat = refCheck ? 1 : 128;
    using Descriptor = typename Sift4bDescTraits::Descriptor;
    using BestMatch = typename Sift4bBruteForceMatchTask::BestMatch;
    const CudaMem<Descriptor, CudaMemType::kManaged> train = allocCudaMem<Descriptor, CudaMemType::kManaged, false>(nbTrainDesc);
    const CudaMem<uint32_t, CudaMemType::kManaged> trainSqrNorm = allocCudaMem<uint32_t, CudaMemType::kManaged, false>(nbTrainDesc);
    std::uniform_int_distribution<uint32_t> dist_desc_word(std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    auto gen_desc = [&](){
        Descriptor result;
        for(auto& w : result)
            w = dist_desc_word(global::random_engine);
        return result;
    };
    std::generate_n(train.get(), nbTrainDesc, gen_desc);
    CudaMem<Descriptor, CudaMemType::kManaged> queries[num_queries];
    CudaMem<uint32_t, CudaMemType::kManaged> queriesSqrNorm[num_queries];
    CudaMem<BestMatch, CudaMemType::kManaged> matches_fwd[num_queries];
    CudaMem<BestMatch, CudaMemType::kManaged> matches_bwd[num_queries];
    for(int i = 0; i < num_queries; i++){
        queries[i] = allocCudaMem<Descriptor, CudaMemType::kManaged, false>(nbQueryDesc);
        std::generate_n(queries[i].get(), nbQueryDesc, gen_desc);
        queriesSqrNorm[i] = allocCudaMem<uint32_t, CudaMemType::kManaged, false>(nbQueryDesc);
        matches_fwd[i] = allocCudaMem<BestMatch, CudaMemType::kManaged>(nbQueryDesc);
        if (bidirectional_match) {
            matches_bwd[i] = allocCudaMem<BestMatch, CudaMemType::kManaged>(nbTrainDesc);
        }
    }

    const cudaStream_t stream = 0;
    const auto event_start = makeCudaEvent(cudaEventDefault);
    const auto event_end = makeCudaEvent(cudaEventDefault);
    const int device = getCudaDevice();

    cudaCheck(cudaMemPrefetchAsync(train.get(), sizeof(Descriptor) * nbTrainDesc, device, stream));
    cudaCheck(cudaMemPrefetchAsync(trainSqrNorm.get(), sizeof(uint32_t) * nbTrainDesc, device, stream));
    launchPreCompSift4bSqrNorm(trainSqrNorm.get(), train.get(), nbTrainDesc, stream);
    for(int i = 0; i < num_queries; i++){
        cudaCheck(cudaMemPrefetchAsync(queries[i].get(), sizeof(Descriptor) * nbQueryDesc, device, stream));
        cudaCheck(cudaMemPrefetchAsync(queriesSqrNorm[i].get(), sizeof(uint32_t) * nbQueryDesc, device, stream));
        launchPreCompSift4bSqrNorm(queriesSqrNorm[i].get(), queries[i].get(), nbQueryDesc, stream);
        cudaCheck(cudaMemPrefetchAsync(matches_fwd[i].get(), sizeof(BestMatch) * nbQueryDesc, device, stream));
        if (bidirectional_match) {
            cudaCheck(cudaMemPrefetchAsync(matches_bwd[i].get(), sizeof(BestMatch) * nbTrainDesc, device, stream));
        }
    }

    std::vector<Sift4bBruteForceMatchTask> tasks(num_queries);
    for (int i = 0; i < num_queries; i++) {
        tasks[i] = Sift4bBruteForceMatchTask{
            cast32u(nbQueryDesc), cast32u(nbTrainDesc),
            queries[i].get(),
            train.get(),
            matches_fwd[i].get(),
            matches_bwd[i].get(),
            queriesSqrNorm[i].get(),
            trainSqrNorm.get()
        };
    }
    // warm up
    for(size_t n = 0; n < nbWarmUpRuns; n++) {
        launchBruteForceMatchIMMA(tasks.data(), tasks.size(), stream);
    }
    cudaCheck(cudaEventRecord(event_start.get(), stream));
    for(size_t n = 0; n < repeat; n++) {
        launchBruteForceMatchIMMA(tasks.data(), tasks.size(), stream);
    }
    cudaCheck(cudaEventRecord(event_end.get(), stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaEventSynchronize(event_start.get()));
    cudaCheck(cudaEventSynchronize(event_end.get()));
	if (!refCheck) {
		float time;
		cudaCheck(cudaEventElapsedTime(&time, event_start.get(), event_end.get()));
		printf("performance: %f FPS\n", repeat * num_queries / time * 1000.f);
	}

    auto getL2Sqr4b = [](const Descriptor& a, const Descriptor& b){
        uint32_t result = 0;
        for(int i = 0; i < (int)sizeof(Descriptor); i++){
            const uint8_t x = reinterpret_cast<const uint8_t*>(&a)[i];
            const uint8_t y = reinterpret_cast<const uint8_t*>(&b)[i];
            result += square(int(x & 0xFu) - int(y & 0xFu));
            result += square(int((x >> 4) & 0xFu) - int((y >> 4) & 0xFu));
        }
        return result;
    };
    auto check_matches = [getL2Sqr4b](
            const Descriptor* query, size_t nbQueryDesc,
            const Descriptor* train, size_t nbTrainDesc,
            const BestMatch* matches){
        const size_t maxNbChecks = 1024;
        const size_t interval = std::max(1ul, nbQueryDesc / maxNbChecks);
        for(size_t i = 0; i < nbQueryDesc; i += interval){
            const auto& m = matches[i];
            ASSERT(m.index < nbTrainDesc);
            if (m.index < nbTrainDesc) {
                const auto refDistance = getL2Sqr4b(query[i], train[m.index]);
                HOPE(m.distance == refDistance);
                for (size_t j = 0; j < nbTrainDesc; j += interval) {
                    HOPE(getL2Sqr4b(query[i], train[j]) >= m.distance);
                }
            }
        }
    };
    if (refCheck) {
        for(int i = 0; i < num_queries; i++){
            check_matches(queries[i].get(), nbQueryDesc, train.get(), nbTrainDesc, matches_fwd[i].get());
            if(bidirectional_match)
                check_matches(train.get(), nbTrainDesc, queries[i].get(), nbQueryDesc, matches_bwd[i].get());
        }
    }
}

TEST(bruteForceMatchTest, dp4aRef)
{
	testBruteForceMatch(false, 897, true);
}

TEST(bruteForceMatchTest, dp4aPerf)
{
	testBruteForceMatch(false, 20000, false);
}

TEST(bruteForceMatchTest, immaRef)
{
	testBruteForceMatch(true, 897, true);
}
TEST(bruteForceMatchTest, immaPerf)
{
	testBruteForceMatch(true, 20000, false);
}

TEST(bruteForceMatchTest, imma4bRef)
{
	testBruteForceMatchIMMA4b(897, true);
}

TEST(bruteForceMatchTest, imma4bPerf)
{
	testBruteForceMatchIMMA4b(20000, false);
}
