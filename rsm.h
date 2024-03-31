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
#include "Types.h"
#include <vector>
#include <variant>

namespace rsm
{
using namespace rsfm;
struct Measurement
{
    uint32_t idxPoint;
    Vec2f location;
};
struct Capture
{
    uint32_t idxCamera;
    uint32_t idxPose;
    std::vector<Measurement> measurements;
    std::string filename;
    std::array<uint32_t, 4> md5sum; // for image file
};

template <uint32_t version>
struct Model;

template <>
struct Model<1>
{
    std::vector<Pose> poses; // Pose::v saved is always in C-fashion, i.e. R * (p - (C + v * (y - height * 0.5f)))
    std::vector<RealCamera> cameras;
    std::vector<rsfm::Vec2<uint32_t>> cameraResolutions;
    std::vector<rsfm::Vec3<Coord>> points;
    std::vector<Color> pointColor;
    
    using Capture = rsm::Capture;
    using Measurement = rsm::Measurement;

    std::vector<Capture> captures;
};

struct RapidSparseModel
{
    static inline constexpr uint64_t magic = 0x974c0afe86e6dfdbUL;
    uint32_t version() const { return static_cast<uint32_t>(model.index()); }
    template <uint32_t version_> const Model<version_>& get() const {return std::get<version_>(model);}
    template <uint32_t version_> Model<version_>& get() {return std::get<version_>(model);}

    std::variant<std::monostate, Model<1>> model;
};

} // namespace rsm