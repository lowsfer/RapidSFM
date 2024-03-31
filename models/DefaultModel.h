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
#include "ModelBase.h"
#include <macros.h>
#include "../ImagePair.h"
#include <boost/pending/disjoint_sets.hpp>
#include <utility>

namespace rsfm
{
class DefaultModel : public ModelBase
{
public:
    using IdxOb = Index;
    using IdxKPt = Index; // [0, nbKPts + nbTiePts) per image
    struct Observation
    {
        ImageHandle hImage;
        IdxKPt idxKPt;
    };
protected:
    void makeIdxObOffsetForImage(ImageHandle hImage, uint32_t nbKPts, uint32_t nbTiePts);
    Observation decodeIdxOb(IdxOb idxOb) const;
    IdxOb encodeIdxOb(const Observation& ob) const;

    const Builder* mBuilder;
private:
    // Note that it's possible that images with smaller handle has larger idxObOffset
    std::unordered_map<ImageHandle, IdxOb, DirectMappingHash<ImageHandle>> mIdxObOffset; // idxOb = mIdxObOffset.at(hImage) + idxKPt
    static constexpr uint32_t idxObBlockSize = 256u;
    std::vector<ImageHandle> mReducedIdxOb2ImgHandle;

    IdxOb mNextIdxObOffset {0}; // of all added images
    std::unordered_map<ImageHandle, std::vector<bool>, DirectMappingHash<ImageHandle>> mObMask; // mask for involved k-points. not used in IncreModel currently
};


} // namespace rsfm
