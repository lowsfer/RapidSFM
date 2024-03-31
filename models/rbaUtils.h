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
#include <RapidBA.h>
#include "../RapidSFM.h"
#include "../Types.h"
#include <cuda_runtime_api.h>

namespace rsfm{

inline rba::IModel::IntriType toRBA(IntriType intriType){
    switch (intriType){
#define INTRI_TYPE_CASE(x) case IntriType::x: return rba::IModel::IntriType::x
        INTRI_TYPE_CASE(kF1);
        INTRI_TYPE_CASE(kF1D2);
        INTRI_TYPE_CASE(kF1D5);
        INTRI_TYPE_CASE(kF1C2D5);
        INTRI_TYPE_CASE(kF2);
        INTRI_TYPE_CASE(kF2C2);
        INTRI_TYPE_CASE(kF2C2D5);
        INTRI_TYPE_CASE(kF2D5);
#undef INTRI_TYPE_CASE
    }
    throw std::runtime_error("fatal error");
}

template <typename T = double3>
inline T toRBA(const Vec3f& p) {return T{p.x, p.y, p.z};}
inline Vec3f fromRBA(const double3& p) {return Vec3f{float(p.x), float(p.y), float(p.z)};}
inline Vec3f fromRBA(const float3& p) {return Vec3f{p.x, p.y, p.z};}
inline float2 toRBA(const Vec2f& p) {return float2{p.x, p.y};}
inline Vec2f fromRBA(const float2& p) {return Vec2f{p.x, p.y};}

inline rba::IModel::Pose<true> toRBA(const Pose& p) {
    return {float4{p.R.x, p.R.y, p.R.z, p.R.w}, double3{p.C.x, p.C.y, p.C.z}, toRBA<float3>(p.v)};
}

inline Pose fromRBA(const rba::IModel::Pose<true>& p) {
    return Pose{Rotation{p.q.w, p.q.x, p.q.y, p.q.z}, Vec3f{float(p.c.x), float(p.c.y), float(p.c.z)}, fromRBA(p.velocity)};
}

inline rba::UniversalIntrinsics toRBA(const RealCamera& cam, uint32_t camResY) {
    return {
        toRBA(cam.pinHole.f), toRBA(cam.pinHole.c),
        cam.k1(), cam.k2(), cam.p1(), cam.p2(), cam.k3(),
		camResY * 0.5f
    };
}

inline RealCamera fromRBA(const rba::UniversalIntrinsics& cam) {
    return {
        {fromRBA(cam.f), fromRBA(cam.c)},
        {cam.k1, cam.k2, cam.p1, cam.p2, cam.k3}
    };
}

} // namespace rsfm
