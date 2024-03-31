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
#include <cstdint>
#include <KArray.h>
#include <stdexcept>
#include <vector>
#include <cuda_utils.h>
namespace rsfm
{

enum class DescElemType {
    kBit,
    kU8,
    kF32,
    kI8,
    kI4,
    kU4
};

constexpr size_t getElemBits(DescElemType elemType){
    switch (elemType) {
        case DescElemType::kBit: return 1;
        case DescElemType::kU8:  return 8;
        case DescElemType::kF32: return 32;
        case DescElemType::kI8:  return 8;
        case DescElemType::kI4:  return 4;
        case DescElemType::kU4:  return 4;
    }
    throw std::logic_error("");
}

enum DistanceStyle
{
    // Hamming-based, lower is better
    kHamming,
    // L2-based, lower is better
    kL2,
    // correlation-based, higher is better. Descriptors should be pre-normalized.
    kDotProd,
};

template <DescElemType descElemType, DistanceStyle distStyle>
using DistanceType =
        std::conditional_t<descElemType == DescElemType::kBit, uint32_t,
        std::conditional_t<descElemType == DescElemType::kU8, uint32_t,
        std::conditional_t<descElemType == DescElemType::kF32, float,
        std::conditional_t<descElemType == DescElemType::kI8, std::conditional_t<distStyle == DistanceStyle::kDotProd, int32_t, uint32_t>,
        std::conditional_t<descElemType == DescElemType::kI4, std::conditional_t<distStyle == DistanceStyle::kDotProd, int32_t, uint32_t>,
        std::conditional_t<descElemType == DescElemType::kU4, uint32_t, void>>>>>>;

template <DescElemType descElemType>
using WordType =
        std::conditional_t<descElemType == DescElemType::kBit, uint32_t,
        std::conditional_t<descElemType == DescElemType::kU8, uint32_t,
        std::conditional_t<descElemType == DescElemType::kF32, float,
        std::conditional_t<descElemType == DescElemType::kI8, int32_t,
        std::conditional_t<descElemType == DescElemType::kI4, int32_t,
        std::conditional_t<descElemType == DescElemType::kU4, uint32_t, void>>>>>>;

template <DescElemType descElemType_, size_t descDims_, DistanceStyle distStyle_>
struct DescTraits
{
    static constexpr int descDims = descDims_;
    static constexpr DescElemType descElemType = descElemType_;    
    static constexpr DistanceStyle distStyle = distStyle_;
    using Word = WordType<descElemType>;
    static constexpr int descWords = getElemBits(descElemType) * descDims / (sizeof(Word) * 8);
    static_assert(sizeof(Word) * 8 * descWords == getElemBits(descElemType) * descDims);
    using Descriptor = cudapp::KArray<Word, descWords>;
    using Distance = DistanceType<descElemType, distStyle>;
};

template <typename Distance>
struct alignas(8) BestMatchImpl
{
    uint32_t index;
    Distance distance;
};

template <DescElemType descElemType, size_t descDims, DistanceStyle distStyle>
struct BruteForceMatchTask {
    using Descriptor = typename DescTraits<descElemType, descDims, distStyle>::Descriptor;
    using Distance = typename DescTraits<descElemType, descDims, distStyle>::Distance;
    uint32_t nbDescA;
    uint32_t nbDescB;
    const Descriptor* __restrict__ descA;
    const Descriptor* __restrict__ descB;

    using BestMatch = BestMatchImpl<Distance>;
    BestMatch* __restrict__ bestMatchA;
    BestMatch* __restrict__ bestMatchB;

    // Used when we use dot product to implement L2. Square norm of each desc can be pre-computed.
    const uint32_t* __restrict__ sqrNormDescA;
    const uint32_t* __restrict__ sqrNormDescB;
};

using SiftDescTraits = DescTraits<DescElemType::kU8, 128, DistanceStyle::kL2>;

using SiftBruteForceMatchTask = BruteForceMatchTask<DescElemType::kU8, 128, DistanceStyle::kL2>;

void launchBruteForceMatchIDP(const SiftBruteForceMatchTask* tasks, size_t nbTasks, cudaStream_t stream);

struct ValidMatch
{
    uint32_t idxQuery;
    uint32_t idxTrain;
    float distance;
};
std::vector<ValidMatch> crossCheckMatches(const typename SiftBruteForceMatchTask::BestMatch* matches, uint32_t nbQueryDesc,
                                   const typename SiftBruteForceMatchTask::BestMatch* matchesBwd, uint32_t nbTrainDesc);
std::vector<ValidMatch> removeMatchConflicts(const typename SiftBruteForceMatchTask::BestMatch* matches, uint32_t nbQueryDesc, uint32_t nbTrainDesc);

using Sift4bDescTraits = DescTraits<DescElemType::kU4, 128, DistanceStyle::kL2>;
using Sift4bBruteForceMatchTask = BruteForceMatchTask<DescElemType::kU4, 128, DistanceStyle::kL2>;
void launchSiftDesc8bTo4b(typename Sift4bDescTraits::Descriptor* dst, const typename SiftDescTraits::Descriptor* src, uint32_t nbDesc, cudaStream_t stream);
void launchPreCompSift4bSqrNorm(uint32_t* sqrNorm, const typename Sift4bDescTraits::Descriptor* src, uint32_t nbDesc, cudaStream_t stream);
void launchBruteForceMatchIMMA(const Sift4bBruteForceMatchTask* tasks, size_t nbTasks, cudaStream_t stream);
void launchPreCompSiftSqrNorm(uint32_t* sqrNorm, const typename SiftDescTraits::Descriptor* src, uint32_t nbDesc, cudaStream_t stream);
void launchBruteForceMatchIMMA(const SiftBruteForceMatchTask* tasks, size_t nbTasks, cudaStream_t stream);

} // namespace rsfm
