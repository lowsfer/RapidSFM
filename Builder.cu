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

#include <cuda_runtime.h>
#include <public_types.h>
#include <cuda_utils.h>

namespace rsfm
{
using LdType = uint4;
__global__ void kernelPickAbstractDesc(
    SiftDescriptor* __restrict__ dst, const SiftDescriptor* __restrict__ src,
    const uint32_t* __restrict__ sampleIndices, uint32_t nbSamples)
{
    
    constexpr uint32_t ldSize = sizeof(LdType);
    constexpr uint32_t grpSize = sizeof(SiftDescriptor) / ldSize;
    assert(blockDim.x * gridDim.x >= grpSize * nbSamples);
    const uint32_t idxThrd = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t idxGrp = idxThrd / grpSize;
    const uint32_t idxInGrp = idxThrd % grpSize;
    if (idxGrp >= nbSamples){
        return;
    }
    const uint32_t idxSample = sampleIndices[idxGrp];
    reinterpret_cast<LdType(*)[grpSize]>(dst)[idxGrp][idxInGrp] =
        reinterpret_cast<const LdType(*)[grpSize]>(src)[idxSample][idxInGrp];
}

cudaError_t cudaPickAbstractDesc(SiftDescriptor* abstractDesc, const SiftDescriptor* descriptors,
    const uint32_t* sampleIndices, uint32_t nbSamples, cudaStream_t stream)
{
    const uint32_t ctaSize = 128u;
    const uint32_t nbCtas = divUp(nbSamples,  ctaSize / (sizeof(SiftDescriptor) / sizeof(LdType)));
    kernelPickAbstractDesc<<<nbCtas, ctaSize, 0, stream>>>(abstractDesc, descriptors, sampleIndices, nbSamples);
    return cudaGetLastError();
}
} // namespace rsfm
