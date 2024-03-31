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
#include <cuda_runtime_api.h>
#include "bruteForceMatch.h"
#include "bruteForceMatch.cuh"
#include <KArray.h>
#include <ptr.h>
#include <ldg.cuh>

namespace rsfm::imma
{
template <typename T, size_t... size>
using KArray = cudapp::KArray<T, size...>;

constexpr int nbBanks = 32;
constexpr int warp_size = 32;

template <DescElemType elemType, int m, int n, int k>
struct MMA
{
    using Word = WordType<elemType>;
    static constexpr int elemsPerWord = sizeof(Word) * 8 / getElemBits(elemType);
    static constexpr int baseMmaM = 8;
    static constexpr int baseMmaN = 8;
    static constexpr int baseMmaK = elemsPerWord * 4;
    static constexpr HW baseDstNbAcc = {1, 2};
    using TileAcc = KArray<Word, baseDstNbAcc.h * m / baseMmaM, baseDstNbAcc.w * n / baseMmaN>;
    using TileA = KArray<Word, k / baseMmaK, m / baseMmaM>;
    using TileB = KArray<Word, k / baseMmaK, n / baseMmaN>;
    __device__ __forceinline__
    static TileAcc run(const TileA& a, const TileB& b, const TileAcc& c);
};

template <> __device__ __forceinline__
KArray<uint32_t, 2, 2> MMA<DescElemType::kU4, 16, 8, 64>::run(
    const KArray<uint32_t, 2, 2>& a, const KArray<uint32_t, 2, 1>& b,
    const KArray<uint32_t, 2, 2>& c)
{
    KArray<uint32_t, 2, 2> d = c;
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1])
        , "r"(b[0][0]), "r"(b[1][0])
        , "r"(c[0][0]), "r"(c[0][1]), "r"(c[1][0]), "r"(c[1][1]));
#else
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[0][0])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[1][0]), "r"(d[1][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[1][0])
        , "r"(b[1][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[1][1])
        , "r"(b[1][0])
        , "r"(d[1][0]), "r"(d[1][1]));
#endif
    return d;
}
template <> __device__ __forceinline__
KArray<int32_t, 2, 2> MMA<DescElemType::kI4, 16, 8, 64>::run(
    const KArray<int32_t, 2, 2>& a, const KArray<int32_t, 2, 1>& b,
    const KArray<int32_t, 2, 2>& c)
{
    KArray<int32_t, 2, 2> d = c;
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1])
        , "r"(b[0][0]), "r"(b[1][0])
        , "r"(c[0][0]), "r"(c[0][1]), "r"(c[1][0]), "r"(c[1][1]));
#else
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[0][0])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[1][0]), "r"(d[1][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[1][0])
        , "r"(b[1][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[1][1])
        , "r"(b[1][0])
        , "r"(d[1][0]), "r"(d[1][1]));
#endif
    return d;
}

template <> __device__ __forceinline__
KArray<uint32_t, 2, 2> MMA<DescElemType::kU4, 16, 8, 32>::run(
    const KArray<uint32_t, 1, 2>& a, const KArray<uint32_t, 1, 1>& b,
    const KArray<uint32_t, 2, 2>& c)
{
    KArray<uint32_t, 2, 2> d = c;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]), "r"(d[1][0]), "r"(d[1][1]));
    return d;
}
template <> __device__ __forceinline__
KArray<int32_t, 2, 2> MMA<DescElemType::kI4, 16, 8, 32>::run(
    const KArray<int32_t, 1, 2>& a, const KArray<int32_t, 1, 1>& b,
    const KArray<int32_t, 2, 2>& c)
{
    KArray<int32_t, 2, 2> d = c;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]), "r"(d[1][0]), "r"(d[1][1]));
    return d;
}

template <> __device__ __forceinline__
KArray<uint32_t, 2, 2> MMA<DescElemType::kU8, 16, 8, 32>::run(
    const KArray<uint32_t, 2, 2>& a, const KArray<uint32_t, 2, 1>& b,
    const KArray<uint32_t, 2, 2>& c)
{
    KArray<uint32_t, 2, 2> d = c;
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1])
        , "r"(b[0][0]), "r"(b[1][0])
        , "r"(c[0][0]), "r"(c[0][1]), "r"(c[1][0]), "r"(c[1][1]));
#else
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[0][0])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[1][0]), "r"(d[1][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[1][0])
        , "r"(b[1][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[1][1])
        , "r"(b[1][0])
        , "r"(d[1][0]), "r"(d[1][1]));
#endif
    return d;
}
template <> __device__ __forceinline__
KArray<int32_t, 2, 2> MMA<DescElemType::kI8, 16, 8, 32>::run(
    const KArray<int32_t, 2, 2>& a, const KArray<int32_t, 2, 1>& b,
    const KArray<int32_t, 2, 2>& c)
{
    KArray<int32_t, 2, 2> d = c;
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[1][0]), "r"(a[1][1])
        , "r"(b[0][0]), "r"(b[1][0])
        , "r"(c[0][0]), "r"(c[0][1]), "r"(c[1][0]), "r"(c[1][1]));
#else
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[0][0])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[1][0]), "r"(d[1][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1])
        : "r"(a[1][0])
        , "r"(b[1][0])
        , "r"(d[0][0]), "r"(d[0][1]));
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%4, %5};\n" 
        : "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[1][1])
        , "r"(b[1][0])
        , "r"(d[1][0]), "r"(d[1][1]));
#endif
    return d;
}

template <> __device__ __forceinline__
KArray<uint32_t, 2, 2> MMA<DescElemType::kU8, 16, 8, 16>::run(
    const KArray<uint32_t, 1, 2>& a, const KArray<uint32_t, 1, 1>& b,
    const KArray<uint32_t, 2, 2>& c)
{
    KArray<uint32_t, 2, 2> d = c;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]), "r"(d[1][0]), "r"(d[1][1]));
    return d;
}
template <> __device__ __forceinline__
KArray<int32_t, 2, 2> MMA<DescElemType::kI8, 16, 8, 16>::run(
    const KArray<int32_t, 1, 2>& a, const KArray<int32_t, 1, 1>& b,
    const KArray<int32_t, 2, 2>& c)
{
    KArray<int32_t, 2, 2> d = c;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n" 
        : "=r"(d[0][0]), "=r"(d[0][1]), "=r"(d[1][0]), "=r"(d[1][1])
        : "r"(a[0][0]), "r"(a[0][1])
        , "r"(b[0][0])
        , "r"(d[0][0]), "r"(d[0][1]), "r"(d[1][0]), "r"(d[1][1]));
    return d;
}


template <int nbBaseMat, typename Word> __device__ __forceinline__
KArray<Word, nbBaseMat> warpLoadMatrix(const Word* ptr);

template <> __device__ __forceinline__
KArray<uint32_t, 4> warpLoadMatrix<4, uint32_t>(const uint32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<uint32_t, 4> r{};
    asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "l"(ptr));
    return r;
}

template <> __device__ __forceinline__
KArray<uint32_t, 2> warpLoadMatrix<2, uint32_t>(const uint32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<uint32_t, 2> r{};
    asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "l"(ptr));
    return r;
}

template <> __device__ __forceinline__
KArray<uint32_t, 1> warpLoadMatrix<1, uint32_t>(const uint32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<uint32_t, 1> r{};
    asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];\n"
        : "=r"(r[0])
        : "l"(ptr));
    return r;
}

template <> __device__ __forceinline__
KArray<int32_t, 4> warpLoadMatrix<4, int32_t>(const int32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<int32_t, 4> r{};
    asm("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "l"(ptr));
    return r;
}

template <> __device__ __forceinline__
KArray<int32_t, 2> warpLoadMatrix<2, int32_t>(const int32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<int32_t, 2> r{};
    asm("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "l"(ptr));
    return r;
}

template <> __device__ __forceinline__
KArray<int32_t, 1> warpLoadMatrix<1, int32_t>(const int32_t* ptr)
{
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
    KArray<int32_t, 1> r{};
    asm("ldmatrix.sync.aligned.m8n8.x1.b16 {%0}, [%1];\n"
        : "=r"(r[0])
        : "l"(ptr));
    return r;
}

template <typename T>
struct NopTransformer
{
	__device__ __forceinline__ T operator()(const T& src) const {return src;}
};

template <DescElemType descElemType, int descDims_, DistanceStyle distStyle_, int nbDesc_>
struct SMemInTile : DescTraits<descElemType, descDims_, distStyle_>
{
    using Traits = DescTraits<descElemType, descDims_, distStyle_>;
    static constexpr int nbDesc = nbDesc_;
    using Word = typename Traits::Word;
    using Descriptor = typename Traits::Descriptor;
    static constexpr HW baseMatShape = {8, 4}; // in words
    static constexpr int nbDescPerLine = nbBanks / Traits::descWords;
    static constexpr int nbLines = nbDesc / nbDescPerLine;
    static constexpr int rowSlidingWords = 4;
    KArray<Word, nbLines, nbBanks> data;

    __device__ __forceinline__
    const Word& at(int idxDesc, int idxWord) const {
        const int r = idxDesc / nbDescPerLine;
        const int c = (Traits::descWords * (idxDesc % nbDescPerLine) + r * rowSlidingWords + idxWord) % nbBanks;
        return data[r][c];
    }
    __device__ __forceinline__
    Word& at(int idxDesc, int idxWord) {
        return const_cast<Word&>(static_cast<const std::decay_t<decltype(*this)>*>(this)->at(idxDesc, idxWord));
    }

    template <int matH, int matW> __device__ __forceinline__
    KArray<Word, matW / baseMatShape.w, matH / baseMatShape.h> warpLoadMatrix(int i, int j) const {
        constexpr HW nbBaseMat = HW{matH, matW} / baseMatShape;
        const auto lane = lane_id();
        // (r, c) are in number of base matrices.
        const auto r = i * nbBaseMat.h + (nbBaseMat.h == 1 ? 0 : lane % 16 / 8);
        const auto c = j * nbBaseMat.w + (nbBaseMat.w == 1 ? 0 : (nbBaseMat.h == 1 ? lane / 8 % 2 : lane / 16));
        const Word* ptr = &this->at(baseMatShape.h * r + lane % baseMatShape.h, baseMatShape.w * c);
        assert(reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0);
        assert(ptr >= data.begin()->begin() && ptr + 16/(sizeof(Word)) <= data.end()->begin());
        const KArray<Word, size_t(nbBaseMat.w * nbBaseMat.h)> tmp = rsfm::imma::warpLoadMatrix<nbBaseMat.w * nbBaseMat.h, Word>(ptr);
        KArray<Word, size_t(nbBaseMat.w), size_t(nbBaseMat.h)> result;
#pragma unroll
        for (int i = 0; i < nbBaseMat.w * nbBaseMat.h; i++) {
            (&result[0][0])[i] = tmp[i];
        }
        return result;
    }

    // ctaDesc is the offseted pointer for this tile, and nbLdDesc is the actual descriptors to load
    template <int ctaSize, bool isFull, typename Transformer = NopTransformer<Word>> __device__ __forceinline__
    void ctaFill(const cudapp::Ptr<const Descriptor>& ctaDesc, const Transformer& transformer = NopTransformer<Word>{}) {
		const auto ptrGen = [&ctaDesc](int32_t idx){
			const auto pDesc = ctaDesc + idx;
			return pDesc.isInBound() ? pDesc.get(): nullptr;
		};
		ctaFillImpl<ctaSize, isFull>(ptrGen, transformer);
    }

	// ctaDesc is the offseted pointer for this tile, and nbLdDesc is the actual descriptors to load
    template <int ctaSize, bool isFull, typename PtrGen, typename Transformer = NopTransformer<Word>> __device__ __forceinline__
    void ctaFillImpl(const PtrGen& ptrGen, const Transformer& transformer = NopTransformer<Word>{}) {
        constexpr int gmemLdWords = 4;
        using GMemLdVec = KArray<Word, gmemLdWords>;
        static constexpr int descGMemVecs = Traits::descWords / gmemLdWords;
        static_assert(Traits::descWords % gmemLdWords == 0);
        using DescVecs = KArray<GMemLdVec, descGMemVecs>;

        const int tid = int(threadIdx.x);
        constexpr int nbLdVecs = Traits::descWords * nbDesc / gmemLdWords;
        static_assert(sizeof(GMemLdVec) * nbLdVecs == sizeof(*this));
        #pragma unroll
        for (int iter = 0; iter < divUp(nbLdVecs, ctaSize); iter++) {
            const int idxVecCta = ctaSize * iter + tid;
            const int idxDescCta = idxVecCta / descGMemVecs;
			if (nbLdVecs % ctaSize != 0 && idxDescCta >= nbDesc) {
				continue;
			}
            const int idxVecInsideDesc = idxVecCta % descGMemVecs;
            const Descriptor* pDesc = ptrGen(idxDescCta);
            GMemLdVec vec;
            if (isFull || pDesc != nullptr) {
                vec = ldg(&reinterpret_cast<const DescVecs&>(*pDesc)[idxVecInsideDesc]);
				#pragma unroll
				for (uint32_t i = 0; i < gmemLdWords; i++) {
					auto& w = vec[i];
					w = transformer(w);
				}
            }
			else {
				vec = GMemLdVec{};
			}

            reinterpret_cast<GMemLdVec&>(this->at(idxDescCta, gmemLdWords * idxVecInsideDesc)) = vec;
        }
    }

	// desc is all the desc for the image. ctaDescIndices is the offseted pointer for this tile, and nbLdDesc is the actual descriptors to load
    template <int ctaSize, bool isFull, typename Transformer = NopTransformer<Word>> __device__ __forceinline__
    void ctaFill(const Descriptor* desc, const cudapp::Ptr<const uint32_t>& ctaDescIndices, const Transformer& transformer = NopTransformer<Word>{}) {
		const auto ptrGen = [&desc, &ctaDescIndices](int32_t idx){
			const auto pIdx = ctaDescIndices + idx;
			return pIdx.isInBound() ? &desc[*pIdx] : nullptr;
		};
		ctaFillImpl<ctaSize, isFull>(ptrGen, transformer);
    }
};

}// namespace rsfm::imma