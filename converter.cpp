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

#include "rsm.hpp"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cmath>
#include "Types.hpp"
namespace fs = std::filesystem;

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>> write(std::ostream& s, T t) {
    s.write((char const*)&t, sizeof(T));
    assert(s);
}

void scaleModel(rsm::Model<1>& m, float scale) {
    for (auto& p : m.poses) {
        p.C = p.C * scale;
        p.v = p.v * scale;
    }
    for (auto& p : m.points) {
        p = p * scale;
    }
}

void saveColmapBin(rsm::Model<1> const& m, fs::path const& p)
{
    bool const hasRollingShutter = std::any_of(m.poses.begin(), m.poses.end(), [](rsm::Pose const& p){
        return p.v.squaredNorm() != 0;
    });
    if (hasRollingShutter) {
        throw std::runtime_error("rolling shutter is not supported in ColMap");
    }
    std::ofstream fout(p/"cameras.bin", std::ios::binary | std::ios::trunc);
    assert(fout);
    write<uint64_t>(fout, m.cameras.size());
    bool const isPinHole = std::all_of(m.cameras.begin(), m.cameras.end(), [](rsfm::RealCamera const& c){
        return c.p1() == 0 && c.p1() == 0 && c.k1() == 0 && c.k2() == 0 && c.k3() == 0;
    });
    if (isPinHole) {
        printf("using pin hole model\n");
    }
    bool const isBundlerModel = std::all_of(m.cameras.begin(), m.cameras.end(), [](rsfm::RealCamera const& c){
        return c.pinHole.f.x == c.pinHole.f.y && c.p1() == 0 && c.p1() == 0 && c.k3() == 0;
    });
    int32_t const camModelId = isPinHole ? 1 : (isBundlerModel ? 3 : 6);
    for (uint32_t i = 0; i < m.cameras.size(); i++) {
        auto const& c = m.cameras.at(i);
        auto const& res = m.cameraResolutions.at(i);
        write<uint32_t>(fout, i);
        write<int32_t>(fout, camModelId);
        write<uint64_t>(fout, res.x);
        write<uint64_t>(fout, res.y);
        if (isPinHole) {
            assert(camModelId == 1);
            for (auto const v : {c.pinHole.f.x, c.pinHole.f.x, c.pinHole.c.x, c.pinHole.c.y}) {
                write<double>(fout, v);
            }
        }
        else if (isBundlerModel){
            assert(camModelId == 3);
            for (auto const v : {c.pinHole.f.x, c.pinHole.c.x, c.pinHole.c.y, c.k1(), c.k2()}) {
                write<double>(fout, v);
            }
        }
        else {
            assert(camModelId == 6);
            for (auto const v : {c.pinHole.f.x, c.pinHole.f.y, c.pinHole.c.x, c.pinHole.c.y, c.k1(), c.k2(), c.p1(), c.p2(), c.k3(), 0.f, 0.f, 0.f}) {
                write<double>(fout, v);
            }
        }
    }
    assert(fout);
    fout.close();

    fout = std::ofstream(p/"images.bin", std::ios::binary | std::ios::trunc);
    write<uint64_t>(fout, m.captures.size());
    using PointIdx = uint32_t;
    using ImgIdx = uint32_t;
    using MeaIdx = uint32_t;
    std::unordered_map<PointIdx, std::pair<std::vector<std::pair<ImgIdx, MeaIdx>>, float>> ptTracks; //!< float is sum of L1 errors.
    for (uint32_t i = 0; i < m.captures.size(); i++) {
        auto const& cap = m.captures.at(i);
        auto const& cam = m.cameras.at(cap.idxCamera);
        auto const& pose = m.poses.at(cap.idxPose);
        write<uint32_t>(fout, i);
        for (float v : {pose.R.w, pose.R.x, pose.R.y, pose.R.z}) {
            write<double>(fout, v);
        }
        auto const t = pose.R * -pose.C;
        for (float v : {t.x, t.y, t.z}) {
            write<double>(fout, v);
        }
        write<uint32_t>(fout, cap.idxCamera);
        auto const filename = fs::path(cap.filename).filename().string();
        fout.write(filename.c_str(), filename.size() + 1);
        write<uint64_t>(fout, cap.measurements.size());
        for (uint32_t j = 0; j < cap.measurements.size(); j++) {
            auto const& ob = cap.measurements.at(j);
            write<double>(fout, ob.location.x);
            write<double>(fout, ob.location.y);
            write<uint64_t>(fout, ob.idxPoint);
            rsfm::Vec2f const normXY = (pose.R * (m.points.at(ob.idxPoint) - pose.C)).homogeneousNormalized();
            auto const proj = (isBundlerModel ? cam.project<2>(normXY) : cam.project<5>(normXY));
            float error = sqrtf((proj - ob.location).squaredNorm());
            auto& entry = ptTracks[ob.idxPoint];
            entry.second += error;
            entry.first.emplace_back(i, j);
        }
    }
    fout.close();

    fout = std::ofstream(p/"points3D.bin", std::ios::binary | std::ios::trunc);
    write<uint64_t>(fout, m.points.size());
    for (uint32_t i = 0; i < m.points.size(); i++) {
        auto const& p = m.points.at(i);
        auto const& c = m.pointColor.at(i);
        write<uint64_t>(fout, i);
        for (float v : {p.x, p.y, p.z}) {
            write<double>(fout, v);
        }
        for (uint8_t v : {c.r, c.g, c.b}) {
            write<uint8_t>(fout, v);
        }
        auto const& track = ptTracks.at(i);
        write<double>(fout, track.second / track.first.size());
        write<uint64_t>(fout, track.first.size());
        for (auto const& [idxImg, idxMea] : track.first){
            write<uint32_t>(fout, idxImg);
            write<uint32_t>(fout, idxMea);
        }
    }
    fout.close();
}

int main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3){
        printf("Usage: converter filename.rsm ${optional_scale}");
        return 1;
    }
    float scale = 1.f;
    if (argc >= 3) {
        std::stringstream ss;
        ss << argv[2];
        ss >> scale;
        assert(ss);
    }
    std::ifstream fin(argv[1], std::ios::binary);
    auto model = rsm::loadRapidSparseModel(fin).get<1>();
    scaleModel(model, scale);
    if (!fs::exists("sparse")) {
        fs::create_directory("sparse");
    }
    assert(fs::is_directory("sparse"));
    saveColmapBin(model, "sparse");
    return 0;
}
