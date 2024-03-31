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
#include <array>
#include <random>
#include <limits>

namespace NonDuplicateSampler{
template<typename IndexType>
class Picker
{
public:
	Picker(uint32_t collection_size, uint32_t seed = std::random_device{}(), uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
	: engine{seed}
    {
        set(collection_size, nbLastFocused, focusWeight);
    }
	IndexType operator()() {
		const auto val = engine() % mExtendedCollectionSize;
		if (val < mNbNonFocused) {
			return static_cast<IndexType>(val);
		}
		return static_cast<IndexType>(mNbNonFocused + (val - mNbNonFocused) / mFocusWeight);
	}
    void set(uint32_t collection_size, uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
    {
        mNbNonFocused = collection_size - nbLastFocused;
	    mExtendedCollectionSize = mNbNonFocused + nbLastFocused * focusWeight;
	    mFocusWeight = focusWeight;
    }
private:
	std::mt19937_64 engine;
	uint32_t mNbNonFocused;
	uint32_t mExtendedCollectionSize;
	uint32_t mFocusWeight;
};
template<typename IndexType, size_t SampleDims>
class TableSampler
{
public:
    typedef std::array<IndexType, SampleDims> indices_type;
    TableSampler(size_t collection_size, uint32_t seed = std::random_device{}(), uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
        : mPicker{static_cast<uint32_t>(collection_size), seed, nbLastFocused, focusWeight}, mMark(collection_size, false) {};
    void set(uint32_t collection_size, uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
    {
        mPicker.set(collection_size, nbLastFocused, focusWeight);
        mMark.resize(collection_size, false);        
    }
    indices_type operator()()
    {
        indices_type indices;
        for(size_t i = 0; i < indices.size(); i++){
            IndexType val;
            do{
                val = mPicker();
            }while(mMark[val]);
            indices[i] = val;
            mMark[val] = true;
        }
        for(auto i : indices)
            mMark[i] = false;
        return indices;
    }
private:
    Picker<IndexType> mPicker;
    std::vector<bool> mMark;
};

template<typename IndexType, size_t SampleDims>
class NaiveSampler
{
public:
    typedef std::array<IndexType, SampleDims> indices_type;
    NaiveSampler(size_t collection_size, uint32_t seed = std::random_device{}(), uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
        : mPicker{static_cast<uint32_t>(collection_size), seed, nbLastFocused, focusWeight}{}
    void set(uint32_t collection_size, uint32_t nbLastFocused = 0, uint32_t focusWeight = 1u)
    {
        mPicker.set(collection_size, nbLastFocused, focusWeight); 
    }
    indices_type operator()()
    {
        indices_type indices;
        for(size_t i = 0; i < indices.size(); i++){
            IndexType val;
            do{
                val = mPicker();
            }while(std::any_of(indices.begin(), indices.begin() + i, [val](IndexType previous){return previous == val;}));
            indices[i] = val;
        }
        return indices;
    }
private:
    Picker<IndexType> mPicker;
};

template<typename IndexType, size_t SampleDims>
using Sampler = typename std::conditional<(SampleDims < 4) , NaiveSampler<IndexType, SampleDims>, TableSampler<IndexType, SampleDims>>::type;
}

template<typename Func, size_t sample_dim, typename index_type = uint32_t, bool sanity_check = true>
std::array<index_type, sample_dim> ransac(const Func& func, const size_t collection_size, const float target_confidence,
	const size_t max_iter = 5000, const uint32_t seed = std::random_device{}(),
	// focusing means when more likely to pick the last nbLastFocused items from the collection, with focusWeight being the ratio to normal items
	const uint32_t nbLastFocused = 0, const uint32_t focusWeight = 1u)
{
    if(collection_size < sample_dim)
        throw std::runtime_error("sampling is not possible");

    typedef std::result_of_t<Func(std::array<index_type, sample_dim>)> result_type;
    typedef std::array<index_type, sample_dim> indices_type;

    NonDuplicateSampler::Sampler<index_type, sample_dim> sampler(collection_size, seed, nbLastFocused, focusWeight);

    result_type best_result = std::numeric_limits<result_type>::min();
    indices_type best_indices;
    std::iota(best_indices.begin(), best_indices.end(), 0);
    size_t required_iters = max_iter;
    size_t num_iters = 0u;
    size_t num_consequitive_insane = 0u;
    do{
        indices_type indices = sampler();
        result_type result = func(indices);
        if(sanity_check)
        {
            if(result <= 0)
            {
                num_consequitive_insane++;
                if(num_consequitive_insane > std::min(128ul, max_iter))
                    break;
                continue;
            }
            else
                num_consequitive_insane = 0u;

        }
        if(result > best_result)
        {
            best_result = result;
            best_indices = indices;
            required_iters = std::min(max_iter,
                                      static_cast<size_t>(std::ceil(log2(1.0 - target_confidence)
                                      / log2(1.f - std::pow(double(best_result) / collection_size, (double)sample_dim)))));
        }
        num_iters++;
    }while(num_iters < required_iters);

    return best_indices;
}
