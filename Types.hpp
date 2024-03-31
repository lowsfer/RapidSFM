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
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/StdVector>
#include <numeric>
namespace rsfm
{

Eigen::Quaternionf toEigen(const Rotation& r);

Rotation fromEigen(const Eigen::Quaternionf& q);

template<typename T>
Eigen::Matrix<T, 3, 1> toEigen(const Vec3<T>& t) {
    return Eigen::Matrix<T, 3, 1>::Map(t.data());
}

template<typename T>
Vec3<T> fromEigen(const Eigen::Matrix<T, 3, 1>& x) {
    return {x[0], x[1], x[2]};
}

template<typename T>
Eigen::Matrix<T, 2, 1> toEigen(const Vec2<T>& t) {
    return Eigen::Matrix<T, 2, 1>::Map(t.data());
}

template<typename T>
Vec2<T> fromEigen(const Eigen::Matrix<T, 2, 1>& x) {
    return {x[0], x[1]};
}

Eigen::Isometry3f toEigen(const Transform& trans);

Transform fromEigen(const Eigen::Isometry3f& trans);

template<typename T>
typename Eigen::Matrix<T, 3, 1>::MapType eigenMap(Vec3<T>& x) {
    return Eigen::Matrix<T, 3, 1>::Map(x.data());
}

template <typename T>
Vec3<T> Vec3<T>::operator-() const{
    return {-x, -y, -z};
}

template<typename T>
using Isometry3 = Eigen::Transform<T, 3, Eigen::Isometry>;
using Isometry3f = Isometry3<float>;

template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;
using Vector3f = Vector3<float>;

template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;
using Vector2f = Vector2<float>;

template<typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>;
using Matrix3f = Matrix3<float>;

template<typename T>
using Affine2 = Eigen::Transform<T, 2, Eigen::AffineCompact>;
using Affine2f = Affine2<float>;

template<typename T>
using Isometry2 = Eigen::Transform<T, 2, Eigen::Isometry>;
using Isometry2f = Isometry2<float>;

template<typename T>
using EigenAlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T = float, bool isInverse = false>
Matrix3<T> toKMat(const PinHoleCameraTemplate<isInverse>& cam) {
    Matrix3<T> kMat;
    kMat << cam.f.x, 0, cam.c.x,
            0, cam.f.y, cam.c.y,
            0, 0, 1;
    return kMat;
}

inline Rotation operator*(const Rotation& R, const Rotation& R0){
    return fromEigen(toEigen(R) * toEigen(R0));
}

inline Vec3<float> operator*(const Rotation& R, const Vec3<float>& x) {
    return fromEigen(toEigen(R) * toEigen(x));
}

inline Isometry3f toEigen(const Transform& trans)
{
    Isometry3f result{toEigen(trans.R)};
    result.translation() = toEigen(trans.t);
    return result;
}

inline Eigen::Affine3f toAffine3f(const Sim3Transform& sim3)
{
    Eigen::Affine3f result{toEigen(sim3.R)};
	result.scale(sim3.scale);
    result.translation() = toEigen(sim3.t);
    return result;
}

inline Rotation fromEigen(const Eigen::Quaternionf& q) {
    return Rotation{q.w(), q.x(), q.y(), q.z()};
}

inline Eigen::Quaternionf toEigen(const Rotation& r) {
    return Eigen::Quaternionf{r.w, r.x, r.y, r.z};
}

// pose1 = trans * pose0
inline Transform operator/(const Pose& pose1, const Pose& pose0) {
    const Rotation R = pose1.R * pose0.R.conjugate();
    const Vec3 t = pose1.R * (pose0.C - pose1.C);
    return Transform{R, t};
}

inline Vec3f Transform::operator*(Vec3f const& p) const {
    return fromEigen(toEigen(*this) * toEigen(p));
}

} // namespace rsfm
