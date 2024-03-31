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

#include <cuda_hint.cuh>
#include <KArray.h>
#include <cuda_runtime.h>
#include "bruteForceMatch.h"
#include <sm_61_intrinsics.h>

namespace rsfm
{
using cudapp::KArray;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
//this allows atomicMax(int,int). atomicMax(float, float) is not natively supported
template <bool isNonNegative>
using OrderedIntType = std::conditional_t<isNonNegative, uint32_t, int32_t>;

template <bool isNonNegative>
__host__ __device__ __forceinline__ OrderedIntType<isNonNegative> float2OrderedInt(float floatVal);
template <> inline uint32_t float2OrderedInt<true>(float floatVal){
    return reinterpret_cast<const uint32_t&>(floatVal);
}
template <> inline int32_t float2OrderedInt<false>(float floatVal){
        const int32_t intVal = reinterpret_cast<const int32_t&>( floatVal );
        return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

template <bool isNonNegative>
__host__ __device__ __forceinline__ float orderedInt2Float(OrderedIntType<isNonNegative> intVal);
template <> inline float orderedInt2Float<true>(uint32_t intVal) {
    return reinterpret_cast<const float&>(intVal);
}
template <> inline float orderedInt2Float<false>(int32_t intVal) {
    const int32_t val = (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
    return reinterpret_cast<const float&>(val);
}
#pragma GCC diagnostic pop

inline constexpr bool isLowerBetter(DistanceStyle style) {
    switch (style) {
        case DistanceStyle::kHamming:
        case DistanceStyle::kL2: return true;
        case DistanceStyle::kDotProd: return false;
    }
    throw std::logic_error("You should never reach here");
}

template <bool isLowerBetter_, typename T>
inline constexpr bool isBetter(const T& a, const T& b) {
    return isLowerBetter_ ? a < b : a > b;
}

template <DescElemType descElemType, DistanceStyle distStyle>
__device__ __forceinline__
DistanceType<descElemType, distStyle> accumulateDistance(const DistanceType<descElemType, distStyle> initVal, const WordType<descElemType> a, const WordType<descElemType> b);

template <>
inline DistanceType<DescElemType::kBit, DistanceStyle::kHamming> accumulateDistance<DescElemType::kBit, DistanceStyle::kHamming>(const DistanceType<DescElemType::kBit, DistanceStyle::kHamming> initVal, const WordType<DescElemType::kBit> a, const WordType<DescElemType::kBit> b) {
    return initVal + __popc(a ^ b);
}

template <>
inline DistanceType<DescElemType::kF32, DistanceStyle::kL2> accumulateDistance<DescElemType::kF32, DistanceStyle::kL2>(const DistanceType<DescElemType::kF32, DistanceStyle::kL2> initVal, const WordType<DescElemType::kF32> a, const WordType<DescElemType::kF32> b) {
    const auto tmp = a - b;
    return initVal + tmp * tmp;
}
template <>
inline DistanceType<DescElemType::kU8, DistanceStyle::kL2> accumulateDistance<DescElemType::kU8, DistanceStyle::kL2>(const DistanceType<DescElemType::kU8, DistanceStyle::kL2> initVal, const WordType<DescElemType::kU8> a, const WordType<DescElemType::kU8> b) {
// Volta and later hardware supports vabsdiff4, but nvcc/ptxas does not support it yet
#define USE_VABSDIFFU4 (__CUDA_ARCH__ < 700)
#if USE_VABSDIFFU4
    const uint32_t diff = __vabsdiffu4(a, b);
    return __dp4a(diff, diff, initVal);
#else
    uint32_t distance = initVal;
    // square(a) and square(b) can be pre-computed externally
    distance = __dp4a(a, a, distance);
    distance = __dp4a(b, b, distance);
    const uint32_t dot_ab = __dp4a(a, b, 0u);
    distance -= dot_ab * 2;
    return distance;
#endif
}

template <>
inline DistanceType<DescElemType::kI8, DistanceStyle::kDotProd> accumulateDistance<DescElemType::kI8, DistanceStyle::kDotProd>(const DistanceType<DescElemType::kI8, DistanceStyle::kDotProd> initVal, const WordType<DescElemType::kI8> a, const WordType<DescElemType::kI8> b) {
    return __dp4a(a, b, initVal);
}

// isNonNegative: whether distance is guaranteed to be non-negative
template <typename Distance, bool isNonNegative, bool isLowerBetter_>
union BestMatchType{
    using Index = uint32_t;
    using OrderedInt = OrderedIntType<isNonNegative>;
    using Raw = typename std::conditional<std::is_signed<OrderedInt>::value, long long, unsigned long long>::type;
    Raw raw;
    static_assert(sizeof(Raw) == 8, "fatal error");
    struct{
        //@important: little endian assumed here!!!
        Index mIndex;
        OrderedInt mOrderedIntDistance;
    };
#ifndef __BYTE_ORDER__
    static_assert(false, "unknown byte order");
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    static_assert(false, "we require little endian");
#endif

    template <bool enabler = true>
    __host__ __device__ __forceinline__
    static std::enable_if_t<enabler && std::is_same<Distance, OrderedInt>::value, OrderedInt> cvt2orderedInt(Distance val) {
        return val;
    }
    template <bool enabler = true>
    __host__ __device__ __forceinline__
    static std::enable_if_t<enabler && std::is_same<Distance, float>::value, OrderedInt> cvt2orderedInt(Distance val) {
        return float2OrderedInt<isNonNegative>(val);
    }

    template <bool enabler = true>
    __host__ __device__ __forceinline__
    static  std::enable_if_t<enabler && std::is_same<Distance, OrderedInt>::value, Distance> cvt2distance(OrderedInt val) {
        return val;
    }
    template <bool enabler = true>
    __host__ __device__ __forceinline__
    static  std::enable_if_t<enabler && std::is_same<Distance, float>::value, Distance> cvt2distance(OrderedInt val) {
        return orderedInt2Float<isNonNegative>(val);
    }

    __host__ __device__ __forceinline__ Distance distance() const{
        return cvt2distance(mOrderedIntDistance);
    }

    __host__ __device__ __forceinline__ Index index() const{
        return mIndex;
    }

    static __host__ __device__ __forceinline__ BestMatchType makeInstance(Distance value, Index index)
    {
        BestMatchType result;
        result.mOrderedIntDistance = cvt2orderedInt(value);
        result.mIndex = index;
        return result;
    }

    __host__ __device__ __forceinline__ void init(Distance value = isLowerBetter_ ? std::numeric_limits<Distance>::max() : std::numeric_limits<Distance>::lowest(),
                                                  Index index = std::numeric_limits<Index>::max()){
        this->raw = makeInstance(value, index).raw;
    }

    __device__ __forceinline__ bool updateRegister(Distance value, Index index){
        const bool need_update = isBetter<isLowerBetter_>(value, distance());
        if(need_update) {
            this->mOrderedIntDistance = cvt2orderedInt(value);
            this->mIndex = index;
        }
        return need_update;
    }

    __device__ __forceinline__ bool updateRam(const BestMatchType& match){
        const Raw result = isLowerBetter_ ? atomicMin(&this->raw, match.raw) : atomicMax(&this->raw, match.raw);
        return result == match.raw;
    }
    __device__ __forceinline__ bool updateRam(Distance value, Index index){
        return updateRam(makeInstance(value, index));
    }
};

// isNonNegative: whether distance is guaranteed to be non-negative
// A smaller BestMatchType with only 32 bits. Index are per CTA local index.
template <bool isDistanceSigned, int nbIdxBits, int nbDistBits, bool isLowerBetter_>
union ReducedBestMatchType{
    using Distance = std::conditional_t<isDistanceSigned, int32_t, uint32_t>;
    static constexpr Distance worstValue = 
        isDistanceSigned
        ? (isLowerBetter_ ? Distance{1u << (nbDistBits - 1)} - 1 : -Distance{1u << (nbDistBits - 1)})
        : Distance{isLowerBetter_ ? (1u << nbDistBits) - 1u : 0u};
    using Index = uint32_t;
    static_assert(std::is_integral<Distance>::value);
    using Raw = typename std::conditional<isDistanceSigned, int32_t, uint32_t>::type;
    Raw raw;
    static_assert(nbDistBits + nbIdxBits == sizeof(raw) * 8);
    struct{
        //@important: little endian assumed here!!!
        Index mIndex        :nbIdxBits;
        Distance mDistance  :nbDistBits;
    };
#ifndef __BYTE_ORDER__
    static_assert(false, "unknown byte order");
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
    static_assert(false, "we require little endian");
#endif

    __host__ __device__ __forceinline__ Distance distance() const{
        return mDistance;
    }

    __host__ __device__ __forceinline__ Index index() const{
        return mIndex;
    }

    static __host__ __device__ __forceinline__
    ReducedBestMatchType makeInstance(Distance value, Index idx) {
        assert(value < (1<<nbDistBits));
        assert(idx < (1<<nbIdxBits));
        ReducedBestMatchType result;
        if (nbIdxBits == 8 && nbDistBits == 24) {
            asm("prmt.b32 %0, %1, %2, 0x6540;" : "=r"(result.raw) : "r"(idx), "r"(value));
        }
        else {
            result.mIndex = idx;
            result.mDistance = value;
        }
        assert(result.distance() == value && result.index() == idx);
        return result;
    }

    __host__ __device__ __forceinline__ void init(Distance value = worstValue,
                                                  Index index = (1u << nbIdxBits) - 1){
        *this = makeInstance(value, index);
    }

    __device__ __forceinline__ bool updateRegister(Raw otherRaw){
        raw = isLowerBetter_ ? std::min(raw, otherRaw) : std::max(raw, otherRaw);
        return raw == otherRaw;
    }

    __device__ __forceinline__ bool updateRegister(const ReducedBestMatchType& other){
        return updateRegister(other.raw);
    }

    __device__ __forceinline__ bool updateRegister(Distance value, Index index){
        return updateRegister(makeInstance(value, index));
    }

    __device__ __forceinline__ bool updateRam(const ReducedBestMatchType& match){
        const Raw result = isLowerBetter_ ? atomicMin(&this->raw, match.raw) : atomicMax(&this->raw, match.raw);
        return result == match.raw;
    }
};

struct HW
{
    int h;
    int w;
    __host__ __device__ __forceinline__
    constexpr HW operator*(const HW other) const {
        return {h * other.h, w * other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator/(const HW other) const {
        return {h / other.h, w / other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator%(const HW other) const {
        return {h % other.h, w % other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator+(const HW other) const {
        return {h + other.h, w + other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator-(const HW other) const {
        return {h - other.h, w - other.w};
    }
    __host__ __device__ __forceinline__
    constexpr int operator[](int idx) const {
        return idx == 0 ? h : w;
    }
};

template <typename Traits, size_t maxNbTasks, uint32_t ctaSizeForInit>
__global__ void kernelInitBruteForceMatch(const std::array<typename Traits::Task, maxNbTasks> tasks) {
    const auto idxTask = blockIdx.z;
    assert(idxTask < tasks.size());
    const auto& task = tasks[idxTask];
    const unsigned x = ctaSizeForInit * blockIdx.x + threadIdx.x;
    if (x < task.nbDescA) {
        reinterpret_cast<typename Traits::BestMatchGlobal&>(task.bestMatchA[x]).init();
    }
    if (task.bestMatchB != nullptr && x < task.nbDescB) {
        reinterpret_cast<typename Traits::BestMatchGlobal&>(task.bestMatchB[x]).init();
    }
}

} // namespace rsfm
