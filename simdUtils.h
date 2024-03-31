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

#include <cstddef>
#include <cstdint>
#include "legacy/aligned_array.hpp"

namespace rsfm
{

template <typename T, size_t vecSize>
using SimdVec = legacy::aligned_array<T, vecSize>;

template <typename T, size_t vecSize, bool needBoundCheck>
inline SimdVec<T, vecSize> loadVec(const T array[], size_t arraySize, uint32_t idx, T filler){
    using Vec = SimdVec<T, vecSize>;
    const T* src = &array[vecSize*idx];
    assert(arraySize > vecSize * idx);
    Vec result;
    if constexpr (needBoundCheck) {
        const size_t nbElems = std::min(vecSize, arraySize - vecSize * idx);
        std::copy_n(src, nbElems, result.data.data());
        std::fill_n(result.data.data() + nbElems, vecSize - nbElems, filler);
    }
    else {
        std::copy_n(src, vecSize, result.data.data());
    }
    return result;
}

template <typename T, size_t vecSize, bool needBoundCheck>
inline void storeVec(T array[], size_t arraySize, const SimdVec<T, vecSize>& __restrict__ vec, uint32_t idx){
    T* dst = &array[vecSize*idx];
    assert(arraySize > vecSize * idx);
    const size_t nbElems = needBoundCheck ? std::min(vecSize, arraySize - vecSize * idx) : vecSize;
    std::copy_n(vec.data.data(), nbElems, dst);
};

} // namespace rsfm
