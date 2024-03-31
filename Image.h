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
#include "fwd.h"
#include <cuda_runtime_api.h>
#include "RapidSFM.h"

namespace rsfm{

struct TiePtMeasurementExt
{
    TiePtHandle hTiePt;
    float x;
    float y;
	uchar3 color; // x,y,z are r,g,b
};
struct Image
{
	float getRollingCenter() const {return height * 0.5f;}

    ImageHandle hImage;
    CameraHandle hCamera;
    PoseHandle hPose;
    std::vector<TiePtMeasurementExt> tiePtMeasurements;
	uint32_t getNbTiePtMeasurements() const {return static_cast<uint32_t>(tiePtMeasurements.size());}

    fs::path file;
    std::array<uint32_t, 4> md5sum; // of file
    int width;
    int height;

    uint32_t nbKPoints; // includes oblique key points
    uint32_t nbNormalKPoints; // First part of keyPoints/descriptors/kptsColor. Extracted from original images.
    using CacheObjKeyType = cudapp::storage::CacheObjKeyType;
    CacheObjKeyType keyPoints;// normal key points first, then oblique key points
    CacheObjKeyType descriptors;
    CacheObjKeyType kptsColor; // uchar3 with x,y,z being r,g,b
    
    // indices to keyPoints/descriptors. Only for BoW database and query, will be cleared after that.
    mutable std::vector<uint32_t> abstract; // Small and cleared soon, so we don't put it in storage manager.
};

Image createImage(Builder* builder, fs::path file, ImageHandle hImage, CameraHandle hCamera, PoseHandle hPose, const std::vector<TiePtMeasurement>& tiePtMeasurements, RapidSift* detector);

} // namespace rsfm
