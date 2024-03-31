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

#include "PnPOptimizer.h"
#include <RapidBA.h>
#include "rbaUtils.h"
#include "../legacy/aligned_array.hpp"
#include <eigen3/Eigen/Dense>
#include <macros.h>
#include "../simdUtils.h"
#include "../SfmUtils.hpp"
#include <iostream>

#ifdef NDEBUG
#pragma GCC optimize("-ffast-math", "-fno-finite-math-only")
#endif
namespace rsfm
{

namespace pnp
{
static constexpr uint32_t simdAlignment = alignof(Eigen::Matrix<float, 32, 1>);
static_assert(simdAlignment == 32); // AVX256
static constexpr uint32_t vecSize = simdAlignment / sizeof(float);
using SimdVecF32 = SimdVec<float, vecSize>;

//point in camera coordinate system and jacobians
struct TransPointDerivative
{
    Vec3<SimdVecF32> position;//point in camera coordinate system

    //jacobians is anti-symmetric
    // deltaGvec*R0*(X - (C0+deltaC)), derivative to deltaGvec and deltaC.    
    // {{0, j01, -j20},
    //  {-j01, 0, j12},
    //  {j20, -j12, 0}}
    SimdVecF32 gvecJac01, gvecJac20, gvecJac12; // jacobian with respect to gvec
    // jacobian for C is identical for all points (equals to -R) and we don't included the data here
    static inline Vec3<Vec3<float>> cJac(const Vec3<Vec3<float>>& R) {
        return -R;
    }
};

template <bool skipDerivative>
inline TransPointDerivative computeTransPointDerivative(const Vec3<Vec3<float>>& R, const Vec3<float>& C, const Vec3<SimdVecF32>& p0)
{
    const Vec3<SimdVecF32> p = p0 - C;
    const Vec3<SimdVecF32> position {
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]
    };
    if (skipDerivative) {
        return {position, {}, {}, {}};
    }
    const SimdVecF32 x0 = p[0] * 2, x1 = p[1] * 2, x2 = p[2] * 2;
    const SimdVecF32 j01 = R[2][0] * x0 + R[2][1] * x1 + R[2][2] * x2;
    const SimdVecF32 j20 = R[1][0] * x0 + R[1][1] * x1 + R[1][2] * x2;
    const SimdVecF32 j12 = R[0][0] * x0 + R[0][1] * x1 + R[0][2] * x2;

    return {position, j01, j20, j12};    
}

struct ProjValueJacobian{
    Vec2<SimdVecF32> value;
    SimdVecF32 transPtJac00, transPtJac11, transPtJac02, transPtJac12; // j01 and j10 are zero
};
template <bool skipDerivative>
inline ProjValueJacobian computeProjValueDerivative(const PinHoleCamera& cam, const Vec3<SimdVecF32>& pt)
{
    const SimdVecF32& x = pt[0];
    const SimdVecF32& y = pt[1];
    const SimdVecF32& z = pt[2];
    const SimdVecF32 z_inv  = 1.f / z;
    const Vec2<SimdVecF32> normXY({x * z_inv, y * z_inv});
    const float fx = cam.f.x, fy = cam.f.y;
    const float cx = cam.c.x, cy = cam.c.y;
    const Vec2<SimdVecF32> pt2d = {normXY.x * fx + cx, normXY.y * fy + cy};
    if (skipDerivative){
        return {pt2d, {}, {}, {}, {}};
    }
    const SimdVecF32 j00 = fx * z_inv;
    const SimdVecF32 j11 = fy * z_inv;
    const SimdVecF32 j02 = -fx * (x * square(z_inv));
    const SimdVecF32 j12 = -fy * (y * square(z_inv));

    return {pt2d, j00, j11, j02, j12};
}

struct ErrorDerivative {
    Vec2<SimdVecF32> error;
    //jacobian for {deltaGvec | deltaC}
    Vec2<Vec3<SimdVecF32>> gvecJac;
    Vec2<Vec3<SimdVecF32>> cJac;
};
template <bool skipDerivative>
inline ErrorDerivative computeErrorDerivative(const PinHoleCamera& cam, const Vec3<Vec3<float>>& R, const Vec3<float>& C,
    const Vec3<SimdVecF32>& p0, const Vec2<SimdVecF32>& observation)
{
    const TransPointDerivative trans = computeTransPointDerivative<skipDerivative>(R, C, p0);

    const ProjValueJacobian proj = computeProjValueDerivative<skipDerivative>(cam, trans.position);

    const Vec2<SimdVecF32> error = observation - proj.value; // Note about x-f(x) and f(x)-x.

    if (skipDerivative) {
        return {error, {}, {}};
    }
    const Vec2<Vec3<SimdVecF32>> gvecJac = {
        {trans.gvecJac20 * proj.transPtJac02, trans.gvecJac01 * proj.transPtJac00 - trans.gvecJac12 * proj.transPtJac02, -trans.gvecJac20 * proj.transPtJac00},
        {-trans.gvecJac01 * proj.transPtJac11 + trans.gvecJac20 * proj.transPtJac12, -trans.gvecJac12 * proj.transPtJac12,  trans.gvecJac12 * proj.transPtJac11}
    };

    const auto transCJac = trans.cJac(R);
    const Vec2<Vec3<SimdVecF32>> cJac = {
        {transCJac[0][0] * proj.transPtJac00 + transCJac[2][0] * proj.transPtJac02, transCJac[0][1] * proj.transPtJac00 + transCJac[2][1] * proj.transPtJac02, transCJac[0][2] * proj.transPtJac00 + transCJac[2][2] * proj.transPtJac02},
        {transCJac[1][0] * proj.transPtJac11 + transCJac[2][0] * proj.transPtJac12, transCJac[1][1] * proj.transPtJac11 + transCJac[2][1] * proj.transPtJac12, transCJac[1][2] * proj.transPtJac11 + transCJac[2][2] * proj.transPtJac12}
    };

    return {error, gvecJac, cJac};
}

//! \returns {weightedOmega, robustChi2}
inline std::pair<SimdVecF32, SimdVecF32> robustifyByHuber(const Vec2<SimdVecF32>& error, const SimdVecF32& omega, const SimdVecF32& delta){
    SimdVecF32 weightedOmega;
    const SimdVecF32 sqrErr = error.squaredNorm() * omega;
    const SimdVecF32 sqrDelta = square(delta);
    const Eigen::Array<bool, vecSize, 1> mask = (sqrErr.data < sqrDelta.data);
    if (mask.all()){
        return {omega, sqrErr};
    }
    else {
        const SimdVecF32 weightedOmega {mask.select(omega.data, omega.data * delta.data * sqrErr.data.rsqrt())};
        const SimdVecF32 robustChi2 {mask.select(sqrErr.data, 2 * sqrErr.data.sqrt() * delta.data - sqrDelta.data)};
        return {weightedOmega, robustChi2};
    }
    DIE("You should never reach here");
}
//! \returns {weightedOmega, robustChi2}
inline std::pair<SimdVecF32, SimdVecF32>  robustifyByDCS(const Vec2<SimdVecF32>& error, const SimdVecF32& omega, const SimdVecF32& delta){
    SimdVecF32 weightedOmega;
    const SimdVecF32 e2 = error.squaredNorm() * omega;
    const SimdVecF32 phi = square(delta);
    const Eigen::Array<bool, vecSize, 1> mask = (e2.data < phi.data);
    if (mask.all()){
        return {omega, e2};
    }
    else {
        const SimdVecF32 weightedOmega {mask.select(omega.data, omega.data * 4 * square(phi).data/square(e2 + phi).data)};
        const SimdVecF32 robustChi2 {mask.select(e2.data, 4*phi.data*e2.data / (phi.data+e2.data) - phi.data)};
        robustChi2.data.isFinite().select(robustChi2.data, 3*phi.data);
        return {weightedOmega, robustChi2};
    }
    DIE("You should never reach here");
}

using AccScalar = double;
static constexpr uint32_t accVecSize = std::min(vecSize, uint32_t(simdAlignment / sizeof(AccScalar)));
using AccVec = SimdVec<AccScalar, accVecSize>;
static_assert(vecSize >= accVecSize && vecSize % accVecSize == 0);

template <bool isFullVec>
static void accumulate(AccVec& acc, SimdVecF32 x, uint32_t nbElems) __attribute__((always_inline));

template <bool isFullVec>
inline void accumulate(AccVec& acc, SimdVecF32 x, uint32_t nbElems) {
    if constexpr(isFullVec) {
        assert(nbElems == vecSize);
    }
    else {
        assert(nbElems < vecSize);
        std::fill(x.data.begin() + nbElems, x.data.end(), 0);
    }
    using Segment = SimdVec<float, accVecSize>::data_type;
    constexpr size_t nbSeg = vecSize / accVecSize;
    #pragma GCC unroll nbSeg
    for (size_t i = 0; i < nbSeg; i++) {
        acc.data += Segment::Map(&x.data[accVecSize * i]).template cast<AccScalar>();
    }
}

template <bool isFullVec, int rows, int cols>
static void accumulate(Eigen::Matrix<AccVec, rows, cols>& acc, const Eigen::Matrix<SimdVecF32, rows, cols>& x, uint32_t nbElems) {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++) {
            accumulate<isFullVec>(acc(i, j), x(i,j), nbElems);
        }
    }
}

template <bool skipDerivative, bool isFullVec>
inline void accumulateHessian(Eigen::Matrix<AccVec, 6, 6>& hessianAcc, Eigen::Matrix<AccVec, 6, 1>& gAcc, AccVec& chi2Acc,
    const ErrorDerivative& errorJac, const SimdVecF32& omega, const SimdVecF32& huber, uint32_t nbElems)
{
    if constexpr(isFullVec) {
        assert(nbElems == vecSize);
    }
    else {
        assert(nbElems < vecSize);
    }
    const auto [weightedOmega, robustChi2] = robustifyByDCS(errorJac.error, omega, huber);
    Eigen::Matrix<SimdVecF32, 2, 6> jacobian;
    using RowMajorMatrix23v = Eigen::Matrix<SimdVecF32, 2, 3, Eigen::RowMajor>;
    jacobian << RowMajorMatrix23v::Map(&errorJac.gvecJac[0][0]), RowMajorMatrix23v::Map(&errorJac.cJac[0][0]);
    if (!skipDerivative) {
        for (int i = 0; i < 6; i++) {
            // hessian is symmetric, we only compute the upper part.
            for (int j = i; j < 6; j++) {
                accumulate<isFullVec>(hessianAcc(i, j), (weightedOmega * jacobian.col(i)).dot(jacobian.col(j)), nbElems);
            }
        }
        accumulate<isFullVec>(gAcc, (jacobian.transpose() * (weightedOmega * toEigen(errorJac.error))).eval(), nbElems);
    }
    accumulate<isFullVec>(chi2Acc, robustChi2, nbElems);
}

template <bool skipDerivative, bool isFullVec>
void accumulateHessian(Eigen::Matrix<AccVec, 6, 6>& hessianAcc, Eigen::Matrix<AccVec, 6, 1>& gAcc, AccVec& chi2Acc,
    const PinHoleCamera& cam, const Vec3<Vec3<float>>& R, const Vec3<float>& C,
    const Vec3<SimdVecF32>& p0, const Vec2<SimdVecF32>& observation,
    const SimdVecF32& omega, const SimdVecF32& huber, uint32_t nbElems)
{
    const ErrorDerivative errJac = computeErrorDerivative<skipDerivative>(cam, R, C, p0, observation);
    accumulateHessian<skipDerivative, isFullVec>(hessianAcc, gAcc, chi2Acc, errJac, omega, huber, nbElems);
}

template <int rows, int cols, bool upperOnly = false>
static inline Eigen::Matrix<AccScalar, rows, cols> reduceVec(const Eigen::Matrix<AccVec, rows, cols>& x)
{
    Eigen::Matrix<AccScalar, rows, cols> result;
    for (int i = 0; i < rows; i++){
        for (int j = upperOnly ? i : 0; j < cols; j++) {
            result(i, j) = x(i, j).sum();
        }
    }
    return result;
}

struct HessianEquation
{
    Eigen::Matrix<AccScalar, 6, 6> hessian;
    Eigen::Matrix<AccScalar, 6, 1> g;
    AccScalar chi2; // not for equation
};
template <bool skipDerivative>
HessianEquation computeHessian(const PinHoleCamera& cam, const Pose& pose,
    const Vec3<const float* __restrict__>& pts3d, const Vec2<const float* __restrict__>& pts2d,
    const float* __restrict__ const omega, const float* __restrict__ const huber, const size_t nbPts)
{
    Eigen::Matrix<AccVec, 6, 6> hessianAcc = Eigen::Matrix<AccVec, 6, 6>::Zero();
    Eigen::Matrix<AccVec, 6, 1> gAcc = Eigen::Matrix<AccVec, 6, 1>::Zero();
    AccVec chi2Acc{0};
    const Vec3<Vec3<float>> R = [](const Rotation& q){
        Vec3<Vec3<float>> r;
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Map(&r[0][0]) = toEigen(q).toRotationMatrix();
        return r;
    }(pose.R);
    const Vec3f& C = pose.C;
    auto processVec = [&]<bool needBoundCheck>(uint32_t idxVec, std::bool_constant<needBoundCheck>){
        const Vec3<SimdVecF32> p0 = {
            loadVec<float, vecSize, needBoundCheck>(pts3d[0], nbPts, idxVec, kNaN),
            loadVec<float, vecSize, needBoundCheck>(pts3d[1], nbPts, idxVec, kNaN),
            loadVec<float, vecSize, needBoundCheck>(pts3d[2], nbPts, idxVec, kNaN)
        };
        const Vec2<SimdVecF32> observation = {
            loadVec<float, vecSize, needBoundCheck>(pts2d[0], nbPts, idxVec, kNaN),
            loadVec<float, vecSize, needBoundCheck>(pts2d[1], nbPts, idxVec, kNaN)
        };
        const SimdVecF32 omegaVec = loadVec<float, vecSize, needBoundCheck>(omega, nbPts, idxVec, kNaN);
        const SimdVecF32 huberVec = loadVec<float, vecSize, needBoundCheck>(huber, nbPts, idxVec, kNaN);
        const size_t nbElems = needBoundCheck ? std::min<uint32_t>(vecSize, nbPts - vecSize * idxVec) : vecSize;
        accumulateHessian<skipDerivative, !needBoundCheck>(hessianAcc, gAcc, chi2Acc, cam, R, C, p0, observation, omegaVec, huberVec, nbElems);
    };
    const uint32_t nbFullVecs = static_cast<uint32_t>(nbPts / vecSize);
    const uint32_t residueVecSize = static_cast<uint32_t>(nbPts % vecSize);
    for (uint32_t i = 0; i < nbFullVecs; i++) {
        processVec(i, std::bool_constant<false>{});
    }
    if (residueVecSize != 0) {
        processVec(nbFullVecs, std::bool_constant<true>{});
    }
    if (skipDerivative) {
        return {Eigen::Matrix<AccScalar, 6, 6>::Constant(kNaN), Eigen::Matrix<AccScalar, 6, 1>::Constant(kNaN), chi2Acc.data.sum()};
    }
    Eigen::Matrix<AccScalar, 6, 6> hessian = reduceVec<6, 6, true>(hessianAcc);
    for (int i = 0; i < 6; i++){
        for (int j = 0; j < i; j++){
            assert((hessianAcc(i, j).data.array() == 0).all());
            hessian(i, j) = hessian(j, i);
        }
    }
    return {hessian, reduceVec(gAcc), chi2Acc.data.sum()};
}

} // namespace pnp

Pose optimizePnP(const PinHoleCamera& cam, const Pose& initPose,
    const Vec3<const float* __restrict__>& pts3d, const Vec2<const float* __restrict__>& pts2d,
    const float* __restrict__ const omega, const float* __restrict__ const huber, const size_t nbPts,
    const int maxNbSteps/* = 60*/, const float initLambda/* = 0.001f*/, const bool verbose/* = false*/)
{
    using namespace pnp;
    const AccScalar thresGraident = 1E-6f;
    const AccScalar thresUpdate = 1E-6f;
    const AccScalar thresChi = 1E-6f;
    const AccScalar thresDeltaChi2 = 0.f;
    const AccScalar maxLambda = 1E2f;
    constexpr int dof = 6;

    Pose pose = initPose;
    AccScalar v = 2.f;
    HessianEquation equation = computeHessian<false>(cam, pose, pts3d, pts2d, omega, huber, nbPts);
    const AccScalar& chi2 = equation.chi2;
    if (!std::isfinite(chi2)) {
        std::cout << "[PnP] chi2 is not finite. Exiting optimization." << std::endl;
    }
    bool stop = false;
    auto requestStop = [&stop](){
        stop = true;
    };
    if ((equation.g.array().abs() < thresGraident).all()) {
        requestStop();
    }
    AccScalar mu = initLambda;
    uint32_t nbAcceptedSteps = 0u;
    uint32_t nbRejectedSteps = 0u;
    for (int k = 0; k < maxNbSteps; k++) {
        if (verbose) {
            std::cout << format("[PnP] LM step %u: chi2 = %e, lambda = %e") % k % chi2 % mu<< std::endl;
        }
        if (stop){
            break;
        }
        const Eigen::Matrix<AccScalar, dof, 1> hessianDiagBackup = equation.hessian.diagonal();
        while (true)
        {
            const Eigen::Matrix<AccScalar, dof, 1> damp = mu * hessianDiagBackup.cwiseMax(1E-6f).cwiseSqrt();
            equation.hessian.diagonal() = hessianDiagBackup + damp;
            const Eigen::Matrix<AccScalar, dof, 1> delta = equation.hessian.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(equation.g);
            if (!delta.allFinite()) {
                std::cout << "[PnP] delta is not finite" << std::endl;
            }
            if (delta.norm() < thresUpdate * (std::sqrt(1 + pose.C.squaredNorm()) + thresUpdate)) {
                requestStop();
            }
            else {
                const Pose newPose {
                    fromEigen(Eigen::Quaternionf{(gvec2mat(delta.template topRows<3>().eval()) * toEigen(pose.R).toRotationMatrix().template cast<AccScalar>()).template cast<float>()}),
                    pose.C + fromEigen(delta.template bottomRows<3>().template cast<float>().eval()),
					zeroVelocity // @fixme: rolling shutter is not yet implemented
                };
                const AccScalar newChi2 = computeHessian<true>(cam, newPose, pts3d, pts2d, omega, huber, nbPts).chi2;
                const AccScalar rho = (chi2 - newChi2) / (delta.dot(damp.cwiseProduct(delta) + equation.g));
                if (rho > 0 && newChi2 < chi2) {
                    if (std::sqrt(chi2) - std::sqrt(newChi2) < thresDeltaChi2 * std::sqrt(chi2)) {
                        requestStop();
                    }
                    pose = newPose;
                    nbAcceptedSteps++;
                    equation = computeHessian<false>(cam, pose, pts3d, pts2d, omega, huber, nbPts);
                    if ((equation.g.array().abs() < thresGraident).all()) {
                        requestStop();
                    }
                    mu = mu * std::max(AccScalar(1)/3, 1 - cube(2*rho - 1));
                    v = 2;
                    break;
                }
                else {
                    if (verbose) {
                        std:: cout << format("[PnP]     Rejected step with chi = %e") % newChi2 << std::endl;
                    }
                    nbRejectedSteps++;
                    mu *= v;
                    v *= 2;
                    if (mu > std::sqrt(hessianDiagBackup.cwiseAbs().maxCoeff()) * maxLambda) {
                        requestStop();
                    }
                }
            }
            if (stop) {
                break;
            }
        }
        if (std::sqrt(chi2) < thresChi) {
            requestStop();
        }
    }
    return pose;
}

} // namespace rsfm
