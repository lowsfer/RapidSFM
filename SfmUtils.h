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
#include <algorithm>
#include <utility>
#include <vector>
#include <cstdint>

#ifdef __GNUC__
    #ifndef __clang__
        #define VECTORIZE_LOOP _Pragma("GCC ivdep")
    #endif
#endif
#ifdef __clang__
    #define VECTORIZE_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#endif


namespace rsfm {

template<typename T>
constexpr T max(T arg0){
    return arg0;
}

template<typename T, typename... Args>
constexpr T max(T arg0, Args... args){
    return std::max(arg0, max(std::forward<Args>(args)...));
}

template <typename Iterator>
std::vector<uint32_t> mask2indices(Iterator beg, Iterator end)
{
    assert(beg < end);
    std::vector<uint32_t> indices;
    for (Iterator iter = beg; iter != end; iter++) {
        if (*iter) {
            indices.push_back(static_cast<uint32_t>(iter - beg));
        }
    }
    return indices;
}

template <typename Container>
std::vector<uint32_t> mask2indices(const Container& container)
{
    return mask2indices(container.begin(), container.end());
}

} // namespace rsfm
