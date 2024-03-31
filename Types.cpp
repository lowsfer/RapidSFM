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

#include "Types.hpp"
#include <eigen3/Eigen/Geometry>
#include <limits>
#include <macros.h>
#include "distortion.h"

namespace rsfm {

constexpr Index kInvalidIndex = std::numeric_limits<Index>::max();

using Eigen::Isometry3f;
using Eigen::Quaternionf;
using Eigen::Vector3f;

Transform fromEigen(const Isometry3f& trans)
{
    return {fromEigen(Quaternionf{trans.rotation()}), fromEigen(trans.translation().eval())};
}

Pose operator*(const Transform& trans, const Pose& pose0) {
    const Rotation R1 = trans.R * pose0.R;
    const Vec3 C1 = pose0.C - R1.conjugate() * trans.t;
    return Pose{R1, C1, zeroVelocity}; // @fixme: check if zeroVelocity is proper everywhere.
}

Transform operator*(const Transform& trans1, const Transform& trans0) {
    const Rotation R = trans1.R * trans0.R;
    const Vec3f t = trans1.R * trans0.t + trans1.t;
    return Transform{R, t};
}

Rotation rvec2quat(const Vec3f &rvec)
{
    const Eigen::Vector3f eigenRVec = toEigen(rvec);
    const float angle = eigenRVec.norm();
    const float angleRcp = 1.f / angle;
    if (!std::isfinite(angleRcp))
    {
        return Rotation::identity();
    }
    const Eigen::AngleAxis angleAxis{angle, eigenRVec * angleRcp};
    const Quaternionf q{angleAxis};
    return fromEigen(q);
}

Transform Transform::inverse() const
{
    return Transform{R.conjugate(), -(R.conjugate() * t)};
}

// normXY is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
template <IntriType intriType>
Vec2f RealCamera::project(const Vec2f& normXY) const {
    return project<getNbDistortParams(intriType)>(normXY);
}
Vec2f RealCamera::project(IntriType intriType, const Vec2f& normXY) const {
#define INTRI_TYPE_CASE(x) case IntriType::x: return project<IntriType::x>(normXY)
    switch (intriType) {
        INTRI_TYPE_CASE(kF1);
        INTRI_TYPE_CASE(kF2);
        INTRI_TYPE_CASE(kF2C2);
        INTRI_TYPE_CASE(kF1D2);
        INTRI_TYPE_CASE(kF1D5);
        INTRI_TYPE_CASE(kF1C2D5);
        INTRI_TYPE_CASE(kF2D5);
        INTRI_TYPE_CASE(kF2C2D5);
    }
#undef INTRI_TYPE_CASE
    DIE("You should never reach here");
}
InverseRealCamera RealCamera::inverse() const {
    return InverseRealCamera{pinHole.inverse(), distortion};
}

// returns is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
template <size_t nbDistortParams>
Vec2f InverseRealCamera::project(const Vec2f& imgPt) const {
    const Vec2f distortNormXY = invPinHole.project(imgPt);
    std::array<float, nbDistortParams> params;
    std::copy_n(distortion.begin(), nbDistortParams, params.begin());
    constexpr uint32_t nbIters = 2;
    return undistort<float, nbDistortParams, nbIters>(distortNormXY, params);
}
template <IntriType intriType>
Vec2f InverseRealCamera::project(const Vec2f& imgPt) const {
    return project<getNbDistortParams(intriType)>(imgPt);
}
#define INSTANTIATE_InverseRealCamera_project(x) \
    template Vec2f InverseRealCamera::project<IntriType::x>(const Vec2f& imgPt) const;
INSTANTIATE_InverseRealCamera_project(kF1)
INSTANTIATE_InverseRealCamera_project(kF2)
INSTANTIATE_InverseRealCamera_project(kF2C2)
INSTANTIATE_InverseRealCamera_project(kF1D2)
INSTANTIATE_InverseRealCamera_project(kF1D5)
INSTANTIATE_InverseRealCamera_project(kF2D5)
INSTANTIATE_InverseRealCamera_project(kF1C2D5)
INSTANTIATE_InverseRealCamera_project(kF2C2D5)
#undef INSTANTIATE_InverseRealCamera_project

Vec2f InverseRealCamera::project(IntriType intriType, const Vec2f& imgPt) const {
#define INTRI_TYPE_CASE(x) case IntriType::x: return project<IntriType::x>(imgPt)
    switch (intriType) {
        INTRI_TYPE_CASE(kF1);
        INTRI_TYPE_CASE(kF2);
        INTRI_TYPE_CASE(kF2C2);
        INTRI_TYPE_CASE(kF1D2);
        INTRI_TYPE_CASE(kF1D5);
        INTRI_TYPE_CASE(kF2D5);
        INTRI_TYPE_CASE(kF1C2D5);
        INTRI_TYPE_CASE(kF2C2D5);
    }
#undef INTRI_TYPE_CASE
    DIE("You should never reach here");
}

std::ostream& operator<<(std::ostream& os, const Pose& p) {
    os << makeFmtStr("R = [%f, %f, %f, %f], C = [%f, %f, %f]", p.R.w, p.R.x, p.R.y, p.R.z, p.C.x, p.C.y, p.C.z);
    return os;
}

//! pixRes is the groud resolution in meters/pixel. It helps unify units of information matrix to pixel^-2
void toInfoMat(float(&dst)[3][3], const Covariance3 & cov, float pixRes) {
    Eigen::Matrix3f covMat;
    covMat << cov.xx, cov.xy, cov.xz,
              cov.xy, cov.yy, cov.yz,
              cov.xz, cov.yz, cov.zz;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Map(&dst[0][0]) = covMat.inverse() * square(pixRes);
}
} // namespace rsfm
