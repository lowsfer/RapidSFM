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
#include "RapidSFM.h"
#include "fwd.h"
#include <vector>
#include "Types.h"

namespace rsfm
{

struct Matches
{
    std::pair<ImageHandle, ImageHandle> images;
    std::vector<std::pair<Index, Index>> kptsMatches;
	std::vector<std::pair<Index, Index>> tiePtMatches; // indices are for Image::tiePtMeasurements
};

struct ImagePair
{
    std::pair<ImageHandle, ImageHandle> images;
    std::vector<std::pair<Index, Index>> kptsMatches;
	std::vector<std::pair<Index, Index>> tiePtMatches; // indices are for Image::tiePtMeasurements

    struct Solution{
        Transform transform;
        std::vector<Index> inliers; // indices to kptsMatches
        float score;
    };
    std::vector<Solution> solutions;
    // max among median of each solutions. @fixme: we may only need one of them
    float maxMedianDepth;
    float minMedianAngle;

    ImagePair inverse() const;
};

ImagePair createImgPair(Builder* builder, ImageHandle hFirst, ImageHandle hSecond,
	std::vector<std::pair<Index, Index>> kptsMatches, std::vector<Pair<Index>> tiePtMatches);

} // namespace rsfm
