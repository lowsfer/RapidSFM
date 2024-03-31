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

#include "DefaultModel.h"
#include "../Image.h"

namespace rsfm
{

DefaultModel::Observation DefaultModel::decodeIdxOb(const DefaultModel::IdxOb idxOb) const
{
    const auto reducedIdxOb = idxOb / idxObBlockSize;
    const ImageHandle hImage = mReducedIdxOb2ImgHandle.at(reducedIdxOb);
    const Observation ob{hImage, idxOb - mIdxObOffset.at(hImage)};
    // assert(encodeIdxOb(ob) == idxOb); // can't enable both this and the assert in encodeIdxOb(), otherwise we have infinite call chain.
    return ob;
}

DefaultModel::IdxOb DefaultModel::encodeIdxOb(const DefaultModel::Observation& ob) const
{
    const IdxOb idxOb = mIdxObOffset.at(ob.hImage) + ob.idxKPt;
    assert(decodeIdxOb(idxOb).hImage == ob.hImage && decodeIdxOb(idxOb).idxKPt == ob.idxKPt);
    return idxOb;
}

void DefaultModel::makeIdxObOffsetForImage(ImageHandle hImage, uint32_t nbKPts, uint32_t nbTiePts)
{
	const uint32_t nbPts = nbKPts + nbTiePts;
    if (mIdxObOffset.count(hImage) != 0) {
        ASSERT(mObMask.at(hImage).size() == nbPts);
        assert(std::all_of(mObMask.at(hImage).begin(), mObMask.at(hImage).end(), [](bool x){return !x;}));
        return;
    }
    ASSERT(mIdxObOffset.try_emplace(hImage, mNextIdxObOffset).second);
    mReducedIdxOb2ImgHandle.insert(mReducedIdxOb2ImgHandle.end(), divUp(nbPts, idxObBlockSize), hImage);
    mNextIdxObOffset += roundUp(nbPts, idxObBlockSize);
    assert(mNextIdxObOffset % idxObBlockSize == 0);
    
    ASSERT(mObMask.emplace(hImage, std::vector<bool>(nbPts, false)).second);
}


} // namespace rsfm
