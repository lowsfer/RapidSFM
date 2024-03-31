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
#include <kmat.h>
namespace rsfm::rmf {
struct alignas(16) PtPair
{
    kmat<float, 2> left;
    kmat<float, 2> right;
};

// votes must be padded to multiples of 4.
void cudaRansacMatchFilter(uint8_t* votes, // length is same as matches
    kmat<float, 2, 3>* affineScratch, // length is cols * rows
    const PtPair* matches,
    uint32_t nbMatches,
    const uint32_t* bounds, // length is cols*rows+1
    uint32_t cols, uint32_t rows, uint32_t nbRansacTests,
    float threshold, bool tryOtherSolutions, cudaStream_t stream);

}
