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
#include "IncreModel.h"
#include <DefaultCacheableObject.h>
#include <RapidSift.h>
#include "../Image.h"
#include <cassert>

namespace rsfm
{

struct IncreModel::CacheEntry
{
    PoseHandle hPose;
    CameraHandle hCamera;
	const Image* image;
    cudapp::storage::AcquiredMemory<const KeyPoint> keyPoints;
    InverseRealCamera camInv;

	auto makePt2dGetter() const {
		return [kpts{keyPoints.data()}, this](IdxKPt idx) {
			if (idx < image->nbKPoints) {
				assert(keyPoints.nbElems() == image->nbKPoints);
				return keyPoints.data()[idx].location;
			}
			else {
				const uint32_t idxTiePt = idx - image->nbKPoints;
				assert(idxTiePt < image->getNbTiePtMeasurements());
				auto& m = image->tiePtMeasurements[idxTiePt];
				return float2{m.x, m.y};
			}
		};
	}

	float2 getPt2d(IdxKPt idx) const {
		return makePt2dGetter()(idx);
	}

	bool isTiePt(IdxKPt idx) const {
		assert(idx < image->nbKPoints + image->getNbTiePtMeasurements());
		return idx >= image->nbKPoints;
	}
};

}