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
#include "rsm.h"
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <iostream>

namespace rsfm
{
template<class Archive> void serialize(Archive & archive, Rotation& r){ archive(r.w, r.x, r.y, r.z); }
template<class Archive, typename Elem> void serialize(Archive & archive, Vec3<Elem>& v){ archive(v.x, v.y, v.z); }
template<class Archive, typename Elem> void serialize(Archive & archive, Vec2<Elem>& v){ archive(v.x, v.y); }
template<class Archive> void serialize(Archive & archive, Color& c){ archive(c.r, c.g, c.b); }
template<class Archive> void serialize(Archive & archive, Pose& p){ archive(p.R, p.C, p.v); }
template<class Archive> void serialize(Archive & archive, PinHoleCamera& p) { archive(p.f, p.c); }
template<class Archive> void serialize(Archive & archive, RealCamera& c){ archive(c.pinHole, c.distortion); }
}
namespace rsm
{
template<class Archive> void serialize(Archive & archive, rsm::Measurement& m){ archive(m.idxPoint, m.location); }
template<class Archive> void serialize(Archive & archive, rsm::Capture& c){ archive(c.idxCamera, c.idxPose, c.measurements, c.filename, c.md5sum); }
template<class Archive> void serialize(Archive & archive, Model<1>& m) { archive(m.poses, m.cameras, m.cameraResolutions, m.points, m.pointColor, m.captures); }

template <typename InStream>
inline RapidSparseModel loadRapidSparseModel(InStream&& stream) {
    static const RapidSparseModel badModel{{std::monostate{}}};
    assert(stream.good() && stream.is_open());
    uint64_t magic = 0;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != RapidSparseModel::magic || !stream.good()) {
        return badModel;
    }
    uint32_t version = 0;
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!stream.good()) {
        return badModel;
    }
    switch (version) {
    case 1: {
        Model<1> model;
        cereal::PortableBinaryInputArchive{stream}(model);
        return RapidSparseModel{{model}};
    }
    default: return badModel;
    }
}
inline void saveRapidSparseModel(const RapidSparseModel& model, std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&RapidSparseModel::magic), sizeof(RapidSparseModel::magic));
    const uint32_t version = model.version();
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    switch (version) {
    case 1: cereal::PortableBinaryOutputArchive{stream}(model.get<1>()); break;
    default: throw std::runtime_error("Invalid model");
    }
    if (!stream.good()) {
        std::runtime_error("bad stream");
    }
}
} // namespace rsfm
