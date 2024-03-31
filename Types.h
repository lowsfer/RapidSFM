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
#include <array>
#include <cstdint>
#include <utility>
#include "RapidSFM.h"
#include <cpp_utils.h>
#include <limits>
#include <cmath>
namespace rba{
enum class ShutterType;
}
namespace rsfm {
using Coord = float;
using Index = uint32_t;
extern const Index kInvalidIndex;

// {w, x, y, z}.
struct Rotation{
    float w, x, y, z;

    float* data() {return &w;}
    const float* data() const {return &w;}
    static constexpr size_t size() {return 4;}
    float& operator[](size_t i) {return data()[i];}
    const float& operator[](size_t i) const {return data()[i];}

    Rotation conjugate() const {return Rotation{w, -x, -y, -z};}
	Rotation normalized() const {
		const float scale = 1.f / std::sqrt(w*w+x*x+y*y+z*z);
		return {w * scale, x * scale, y * scale, z * scale};
	}
    static constexpr Rotation identity() {return Rotation{1, 0, 0, 0};}
};

template <typename T>
struct Vec2 {
    T x, y;

    __host__ __device__ inline
    T* data() {return &x;}
    __host__ __device__ inline
    const T* data() const {return &x;}
    static constexpr size_t size() {return 2;}
    __host__ __device__ inline
    T& operator[](size_t i) {return data()[i];}
    __host__ __device__ inline
    const T& operator[](size_t i) const {return data()[i];}
    __host__ __device__ inline
    Vec2<T> operator-() const {return {-x, -y};}
    template <typename T2>
    __host__ __device__ inline
    auto operator-(const Vec2<T2> &rhs) const -> Vec2<decltype(std::declval<T>() - std::declval<T2>())> {
        return {x - rhs.x, y - rhs.y};
    }

    template <typename T2>
    __host__ __device__ inline
    auto operator+(const Vec2<T2> &rhs) const -> Vec2<decltype(std::declval<T>() + std::declval<T2>())> {
        return {x + rhs.x, y + rhs.y};
    }
    __host__ __device__ inline
    Vec2<T> operator*(const Vec2<T>& rhs) const {
        return {x * rhs.x, y * rhs.y};
    }
    __host__ __device__ inline
    Vec2<T> operator/(const Vec2<T>& rhs) const {
        return {x / rhs.x, y / rhs.y};
    }
    __host__ __device__ inline
    Vec2<T> operator-(const T& rhs) const {
        return {x - rhs, y - rhs};
    }
    __host__ __device__ inline
    Vec2<T> operator+(const T& rhs) const {
        return {x + rhs, y + rhs};
    }
    template <typename T2>
    __host__ __device__ inline
    auto operator*(const T2& rhs) const -> Vec2<decltype(std::declval<T>() * std::declval<T2>())> {
        return {x * rhs, y * rhs};
    }
    __host__ __device__ inline
    bool operator==(const Vec2<T>& src) const {
        return src.x == x && src.y == y;
    }
    __host__ __device__ inline
    T squaredNorm() const {return square(x) + square(y);}
};

using Vec2f = Vec2<float>;

template <typename T>
struct Vec3{
    T x, y, z;

    T* data() {return &x;}
    const T* data() const {return &x;}
    static constexpr size_t size() {return 3;}

    T& operator[](size_t i) {return data()[i];}
    const T& operator[](size_t i) const {return data()[i];}

    Vec3<T> operator-() const;

    template <typename T2>
    auto operator-(const Vec3<T2> &rhs) const -> Vec3<decltype(std::declval<T>() - std::declval<T2>())> {
        return {x - rhs.x, y - rhs.y, z - rhs.z};
    }

    template <typename T2>
    auto operator+(const Vec3<T2> &rhs) const -> Vec3<decltype(std::declval<T>() + std::declval<T2>())> {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }
    template <typename T2>
    auto operator*(const T2& rhs) const -> Vec3<decltype(std::declval<T>() * std::declval<T2>())> {
        return {x * rhs, y * rhs, z * rhs};
    }

    T squaredNorm() const {return x*x + y*y + z*z;}
    bool operator==(const Vec3<T>& src) const {
        return src.x == x && src.y == y && src.z == z;
    }

    Vec2<T> homogeneousNormalized() const {
        const T zRcp = T(1) / z;
        return {x * zRcp, y * zRcp};
    }

    T dot(const Vec3<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
};

// template <typename T1, typename T2>
// auto operator*(const Vec3<Vec3<T1>>& A, const Vec3<T2>& B) -> Vec3<decltype(std::declval<T1>() * std::declval<T2>())> {
//     return {
//         A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2],
//         A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2],
//         A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2]
//     };
// }

using Vec3f = Vec3<float>;


Rotation rvec2quat(const Vec3f& rvec);

inline constexpr Vec3f zeroVelocity{0.f, 0.f, 0.f};

Rotation operator*(const Rotation& R, const Rotation& R0);
Vec3<float> operator*(const Rotation& R, const Vec3<float>& t);

struct Pose
{
    Rotation R;
    Vec3<Coord> C;
	// velocity for rolling shutter. Camera moving distance per rolling shutter line scan time.
	// In T or C fashion depending on shutter type. See rba::ShutterType.
	// When saved to rsm file, it's always converted to C-fashion
	Vec3<float> v;

    // rolling Center is (width * 0.5f). rollVGlb==true means v is in C-fashion and applied on C.
	const Vec3<Coord> getRollingC(bool rollVGlb, float y, float rollingCenter) const {
		return rollVGlb ? C + v * (y - rollingCenter) : C + R.conjugate() * v * (y - rollingCenter);
	}

	const Pose withVelocity(Vec3<float> velocity) const {
		return {R, C, velocity};
	}

    static Pose identity() {
        return Pose{Rotation::identity(), Vec3<Coord>{0.f, 0.f, 0.f}, zeroVelocity};
    }

    // rollingCenter is typically at the vertical center of the image (not optical center!).
	const Vec3f transform(const Vec3f& p0, rba::ShutterType shutter, float y, float rollingCenter) const;
};

std::ostream& operator<<(std::ostream& os, const Pose& p);

struct Transform
{
    Rotation R;
    Vec3<float> t;

    Transform inverse() const;
    Pose toPose() const {
        return Pose{R, -(R.conjugate() * t), zeroVelocity};
    }
    static Transform fromPose(const Pose& p) {
        return Transform{p.R, p.R * -p.C};
    }
    // rolling Center is (width * 0.5f)
	static Transform fromRollingPose(const Pose& p, bool rollVGlb, float y, float rollingCenter) {
        return Transform{p.R, p.R * -p.getRollingC(rollVGlb, y, rollingCenter)};
    }

    Vec3f operator*(Vec3f const& p) const;
};

Pose operator*(const Transform& trans, const Pose& pose0);
// pose1 = trans * pose0
Transform operator/(const Pose& pose1, const Pose& pose0);
Transform operator*(const Transform& trans1, const Transform& trans0);

struct Sim3Transform
{
	Rotation R;
	float scale;
	Vec3<float> t;
};

template <bool isInverse>
struct PinHoleCameraTemplate
{
    Vec2f f;
    Vec2f c;

    __host__ __device__ inline
    PinHoleCameraTemplate<!isInverse> inverse() const {
        const Vec2f fInv = {1.f/f.x, 1.f/f.y};
        const Vec2f cInv = {-c.x * fInv.x, -c.y * fInv.y};
        return {fInv, cInv};
    }

    // normXY <-> pixel coordinates
    // normXY is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
    __host__ __device__ inline
    Vec2f project(const Vec2f& pt2d) const {
        return Vec2f{f.x * pt2d.x + c.x, f.y * pt2d.y + c.y};
    }
};

using PinHoleCamera = PinHoleCameraTemplate<false>;
using InversePinHoleCamera = PinHoleCameraTemplate<true>;

constexpr uint32_t getNbDistortParams(IntriType intriType)
{
    switch (intriType) {
    case IntriType::kF1:        return 0;
    case IntriType::kF2:        return 0;
    case IntriType::kF2C2:      return 0;
    case IntriType::kF1D2:      return 2;
    case IntriType::kF1D5:    return 5;
    case IntriType::kF2D5:    return 5;
    case IntriType::kF1C2D5:    return 5;
    case IntriType::kF2C2D5:    return 5;
    }
    return std::numeric_limits<uint32_t>::max();
}

struct InverseRealCamera
{
    InversePinHoleCamera invPinHole; // inversed
    std::array<float, 5> distortion;

    // returns is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
    template <size_t nbDistortParams>
    Vec2f project(const Vec2f& imgPt) const;
    template <IntriType intriType>
    Vec2f project(const Vec2f& imgPt) const;
    Vec2f project(IntriType intriType, const Vec2f& imgPt) const;
};

struct RealCamera
{
    PinHoleCamera pinHole;
    std::array<float, 5> distortion;
    __host__ __device__ inline
    float k1() const {return distortion[0];}
    __host__ __device__ inline
    float k2() const {return distortion[1];}
    __host__ __device__ inline
    float p1() const {return distortion[2];}
    __host__ __device__ inline
    float p2() const {return distortion[3];}
    __host__ __device__ inline
    float k3() const {return distortion[4];}

    // normXY is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
    template <size_t nbDistortParams>
    __host__ __device__ inline
    Vec2f project(const Vec2f& normXY) const;
    template <IntriType intriType>
    Vec2f project(const Vec2f& normXY) const;
    Vec2f project(IntriType intriType, const Vec2f& normXY) const;
    // only apply distortion, not projection.
    template <size_t nbDistortParams>
    __host__ __device__ inline
    Vec2f distort(const Vec2f& normXY) const;

    bool isPinHole() const {return std::all_of(distortion.begin(), distortion.end(), [](float x){return x == 0;});}

    InverseRealCamera inverse() const;
};
// normXY is the homogeneous normalized 3d coordinates, i.e. x/z and y/z, in camera coordinate system
template <size_t nbDistortParams>
__host__ __device__ inline
Vec2f distort(const float* params, const Vec2f& normXY) {
    Vec2f newXY = normXY;
    const float nx = normXY.x, ny = normXY.y;
    const float r2 = nx*nx + ny*ny;
    float radial = 1.f;
    static_assert(nbDistortParams < 6 && nbDistortParams != 3);
    if (nbDistortParams >= 1) radial += params[0] * r2;
    if (nbDistortParams >= 2) radial += params[1] * (r2 * r2);
    if (nbDistortParams >= 5) radial += params[4] * (r2 * r2 * r2);
    if (nbDistortParams != 0) newXY = newXY * radial;
    if (nbDistortParams >= 4) {
        const auto& p1 = params[2];
        const auto& p2 = params[3];
        newXY = newXY + Vec2f{2*p1*(nx*ny) + p2*(r2 + 2*(nx*nx)), 2*p2*(nx*ny) + p1*(r2 + 2*(ny*ny))};
    }
    return newXY;
}

template <size_t nbDistortParams>
__host__ __device__ inline
Vec2f RealCamera::distort(const Vec2f& normXY) const {
    return rsfm::distort<nbDistortParams>(distortion.data(), normXY);
}
template <size_t nbDistortParams>
__host__ __device__ inline
Vec2f RealCamera::project(const Vec2f& normXY) const {
    return pinHole.project(distort<nbDistortParams>(normXY));
}

struct Color{
    uint8_t r, g, b;
};
template <typename T>
using Pair = std::pair<T, T>;

struct LocCtrl{
    Vec3<Coord> loc;
    float huber;
    Covariance3 cov;
};

//! pixRes is the groud resolution in meters
void toInfoMat(float(&dst)[3][3], const Covariance3 & cov, float pixRes);

} // namespace rsfm
