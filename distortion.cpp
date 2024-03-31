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

#ifdef NDEBUG
#pragma GCC optimize("-ffast-math", "-fno-finite-math-only")
#endif

#include "distortion.h"
#include <type_traits>
#include <cstddef>
#include "Types.h"
#include <macros.h>
#include "legacy/aligned_array.hpp"
#include <cpp_utils.h>
#include <initializer_list>
#include "SfmUtils.hpp"
#include "simdUtils.h"
#include <sstream>
#include <iostream>

namespace rsfm
{
static constexpr uint32_t simdAlignment = alignof(Eigen::Array<float, 32, 1>);
static_assert(simdAlignment == 32); // AVX256
namespace sym {
struct Zero {
    template <typename T>
    constexpr explicit operator T() const {return T{0};}
    bool operator==(const Zero&) const {return true;}
};

template<typename T>
constexpr Zero operator*(const Zero &, const T &) { return Zero{}; }

template<typename T, std::enable_if_t<!std::is_same<T,Zero>::value, int> = 0>
constexpr Zero operator*(const T &, const Zero &) { return Zero{}; }

template<typename T>
constexpr T operator+(const Zero &, const T &b) { return b; }

template<typename T, std::enable_if_t<!std::is_same<T,Zero>::value, int> = 0>
constexpr T operator+(const T &a, const Zero &) { return a; }

template <typename T>
constexpr bool isSymZero() {return std::is_same_v<std::decay_t<T>, Zero>;}
} // namespace sym

// constexpr uint32_t nbParams = 4;
template <uint32_t nbParams>
struct DistortParams{
    DistortParams() = default;
    DistortParams(const std::array<float, nbParams>& init) :params{init}{}
    static_assert(nbParams <= 5, "Too many parameters");
    constexpr static uint32_t DoF = nbParams;
    // k1, k2, p1, p2, k3
    std::array<float, DoF> params;
#define DEFINE_ACCESSOR(name, index) \
    template <bool enabler = true, std::enable_if_t<enabler&&(DoF>index), int> = 0> \
    float name() const { return params[index];} \
    template <bool enabler = true, std::enable_if_t<enabler&&(DoF<=index), int> = 0> \
    sym::Zero name() const { return sym::Zero{};}
    DEFINE_ACCESSOR(k1, 0)
    DEFINE_ACCESSOR(k2, 1)
    DEFINE_ACCESSOR(p1, 2)
    DEFINE_ACCESSOR(p2, 3)
    DEFINE_ACCESSOR(k3, 4)
#undef DEFINE_ACCESSOR

    template <typename T>
    Vec2<T> computeValue(const Vec2<T>& normXY) const
    {
        const auto& k1 = this->k1();
        const auto& k2 = this->k2();
        const auto& p1 = this->p1();
        const auto& p2 = this->p2();
        const auto& k3 = this->k3();
        const T r2 = square(normXY.x) + square(normXY.y);
        const T x0 = normXY[0], x1 = normXY[1];
        const T scale = 1.f + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2);
        Vec2<T> value = normXY * scale;
        if constexpr (!sym::isSymZero<decltype(p1)>() || !sym::isSymZero<decltype(p2)>()) {
            value = value + Vec2<T>{
                2*p1*(x0*x1) + p2*r2 + 2*p2*square(x0),
                2*p2*(x0*x1) + p1*r2 + 2*p1*square(x1)
            };
        }
        return value;
    }

    template <typename T>
    std::pair<Vec2<T>, Vec2<Vec2<T>>> computeValueJacobian(const Vec2<T>& normXY) const
    {
        const auto& k1 = this->k1();
        const auto& k2 = this->k2();
        const auto& p1 = this->p1();
        const auto& p2 = this->p2();
        const auto& k3 = this->k3();
        const T r2 = square(normXY.x) + square(normXY.y);
        const T x0 = normXY[0], x1 = normXY[1];
        const T scale = 1.f + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2);
        Vec2<T> value = normXY * scale;
        if constexpr (!sym::isSymZero<decltype(p1)>() || !sym::isSymZero<decltype(p2)>()) {
            value = value + Vec2<T>{
                2*p1*(x0*x1) + p2*r2 + 2*p2*square(x0),
                2*p2*(x0*x1) + p1*r2 + 2*p1*square(x1)
            };
        }
        // assert(value == computeValue(normXY));

        const auto scaleJacR2 = k1 + 2.f*k2*r2 + 3.f*k3*(r2*r2);
        const auto j01 = T{2.f*x0*x1*scaleJacR2 + 2.f*p1*x0 + 2.f*p2*x1};
        const Vec2<Vec2<T>> jacobian {
            {scale + 2.f*scaleJacR2*square(x0) + 2.f*p1*x1 + 6.f*p2*x0, j01},
            {j01, scale + 2.f*scaleJacR2*square(x1) + 6.f*p1*x1 + 2.f*p2*x0}
        };
        return {value, jacobian};
    }
};

// using T = double; constexpr uint32_t nbParams = 3;
template <typename T, uint32_t nbParams, uint32_t nbIters> // typically 2 iterations are enough
Vec2<T> undistort(const Vec2<T>& __restrict__ distortedNormXY, const std::array<float, nbParams>& __restrict__ params)
{
    const DistortParams<nbParams> distortion(params);
    Vec2<T> solution = distortedNormXY;
#ifndef NDEBUG
#define CHECK_CONVERGE 1
#else
#define CHECK_CONVERGE 0
#endif
#if CHECK_CONVERGE
    std::array<std::pair<Vec2<T>, Vec2<T>>, nbIters + 1> history; // {error, solution}
#endif
    for (uint32_t i = 0; i < nbIters; i++) {
        const auto [value, jacobian] = distortion.computeValueJacobian(solution);
        const auto error = distortedNormXY - value;
#if CHECK_CONVERGE
        history.at(i) = {error, solution};
#endif
        // printf("error: %f %f\n", error.x, error.y);
        const auto& j = jacobian;
        const auto h01 = j[0][0]*j[0][1]+j[1][0]*j[1][1];
        const auto& h10 = h01;
        const Vec2<Vec2<T>> hessian = {
            {square(j[0][0]) + square(j[1][0]), h01},
            {h10, square(j[0][1]) + square(j[1][1])}
        };
        const Vec2<T> g = {
            j[0][0] * error[0] + j[1][0] * error[1],
            j[0][1] * error[0] + j[1][1] * error[1]
        };
        // Solve hessian * delta = error;
        Vec2<T> delta;
        Eigen::Matrix<T, 2, 1>::Map(delta.data()) = solveGaussElim(
            Eigen::Matrix<T, 2, 2, Eigen::RowMajor>::Map(&hessian[0][0]),
            Eigen::Matrix<T, 2, 1>::Map(g.data()));
        solution = solution + delta;
    }
#if CHECK_CONVERGE
    const auto finalError = distortedNormXY - distortion.computeValue(solution);
    history.at(nbIters) = {finalError, solution};
    bool hasBadSolution = false;
    if constexpr (std::is_same_v<std::decay_t<T>, float>) {
        hasBadSolution = !(finalError.squaredNorm() < 1E-10f);
    }
    else {
        hasBadSolution = !(finalError.squaredNorm().data < 1E-10f).all();
    }
    if (hasBadSolution) {
        auto printVec2 = [](const Vec2<T>& src) {
            std::stringstream ss;
            if constexpr (std::is_same_v<std::decay_t<decltype(src.x)>, float>) {
                ss << "{" << src.x << "," << src.y << "}";
            }
            else {
                ss << "{";
                for (int32_t i = 0; i < src.x.data.size(); i++) {
                    ss << "{" << src.x.data[i] << "," << src.y.data[i] << (i+1 == src.x.data.size() ? "}," : "}");
                }
                ss << "}";
            }
            return ss.str();
        };
        std::stringstream ss;
        ss << "**********************************\n";
        ss << "Target: " << printVec2(distortedNormXY) << "\n";
        for (unsigned i = 0; i < history.size(); i++) {
            ss << "  error: " << printVec2(history.at(i).first) << "\n  solution: " << printVec2(history.at(i).second) << "\n";
        }
        ss << "++++++++++++++++++++++++++++++++++\n";
        std::cout << ss.str() << std::flush;
    }
#endif
#undef CHECK_CONVERGE
    return solution;
}
template Vec2<float> undistort<float, 0, 2>(const Vec2<float>&, const std::array<float, 0>&);
template Vec2<float> undistort<float, 1, 2>(const Vec2<float>&, const std::array<float, 1>&);
template Vec2<float> undistort<float, 2, 2>(const Vec2<float>&, const std::array<float, 2>&);
template Vec2<float> undistort<float, 4, 2>(const Vec2<float>&, const std::array<float, 4>&);
template Vec2<float> undistort<float, 5, 2>(const Vec2<float>&, const std::array<float, 5>&);
template Vec2<float> undistort<float, 0, 3>(const Vec2<float>&, const std::array<float, 0>&);
template Vec2<float> undistort<float, 1, 3>(const Vec2<float>&, const std::array<float, 1>&);
template Vec2<float> undistort<float, 2, 3>(const Vec2<float>&, const std::array<float, 2>&);
template Vec2<float> undistort<float, 4, 3>(const Vec2<float>&, const std::array<float, 4>&);
template Vec2<float> undistort<float, 5, 3>(const Vec2<float>&, const std::array<float, 5>&);

template <uint32_t nbParams, uint32_t vecSize = simdAlignment / sizeof(float), uint32_t nbIters = 2>
void undistortInPlaceImpl(float px[], float py[], size_t nbPts, const PinHoleCamera& cam, const std::array<float, nbParams>& distortion)
{
    using Vec = legacy::aligned_array<float, vecSize>;
    static_assert(sizeof(Vec) == sizeof(float) * vecSize);
    static_assert(sizeof(Vec) == alignof(Vec));
    const auto project = [](const Vec2<Vec>& p, const auto& cam) -> Vec2<Vec> {
        return {p.x * cam.f.x + cam.c.x, p.y * cam.f.y + cam.c.y};
    };
    const InversePinHoleCamera camInv = cam.inverse();
    auto processVec = [&]<bool needBoundCheck>(uint32_t idxVec, std::bool_constant<needBoundCheck>) mutable{
        Vec2<Vec> p = {
            loadVec<float, vecSize, needBoundCheck>(px, nbPts, idxVec, 0),
            loadVec<float, vecSize, needBoundCheck>(py, nbPts, idxVec, 0),
        };
        p = project(p, camInv);
        undistort<Vec, nbParams, nbIters>(p, distortion);
        p = project(p, cam);
        storeVec<float, vecSize, needBoundCheck>(px, nbPts, p.x, idxVec);
        storeVec<float, vecSize, needBoundCheck>(py, nbPts, p.y, idxVec);
    };
    for (uint32_t idxVec = 0; idxVec < nbPts / vecSize; idxVec++) {
        processVec(idxVec, std::bool_constant<false>{});
    }
    if (nbPts % vecSize != 0) {
        processVec(nbPts / vecSize, std::bool_constant<true>{});
    }
}

// px, py are the 2d image coordinates in pixels
void undistortInPlace(float px[], float py[], size_t nbPts, const RealCamera& camera)
{
    ASSERT(reinterpret_cast<std::uintptr_t>(px) % simdAlignment == 0 && reinterpret_cast<std::uintptr_t>(py) % simdAlignment == 0);
    const auto& d = camera.distortion;
    const size_t nbParams = d.rend() - std::find_if(d.rbegin(), d.rend(), [](float x){return x != 0;});

    const auto vecSize = simdAlignment / sizeof(float);
    const uint32_t nbIters = 2; // typically 2 is enough

    switch (nbParams)
    {
    case 0: break;
    case 1: undistortInPlaceImpl<1, vecSize, nbIters>(px, py, nbPts, camera.pinHole, {{d[0]}}); break;
    case 2: undistortInPlaceImpl<2, vecSize, nbIters>(px, py, nbPts, camera.pinHole, {{d[0], d[1]}}); break;
    case 3: undistortInPlaceImpl<3, vecSize, nbIters>(px, py, nbPts, camera.pinHole, {{d[0], d[1], d[2]}}); break;
    case 4: undistortInPlaceImpl<4, vecSize, nbIters>(px, py, nbPts, camera.pinHole, {{d[0], d[1], d[2], d[3]}}); break;
    case 5: undistortInPlaceImpl<5, vecSize, nbIters>(px, py, nbPts, camera.pinHole, {{d[0], d[1], d[2], d[3], d[4]}}); break;
    default: DIE("fatal error");
    }
}

} // namespace rsfm
