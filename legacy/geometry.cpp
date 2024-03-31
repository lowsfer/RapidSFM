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

//
// Created by yao on 21/08/17.
//
#include "geometry.hpp"
#include <cpp_utils.h>
#include <array>
#include "../Types.hpp"
#include <lambdatwist/p3p.h>

namespace rsfm
{
Eigen::Array<bool, Eigen::Dynamic, 1> checkAffine2Transformation(
    const Affine2f &trans, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold)
{
    const auto& mat = trans.matrix();

    const float scale = std::sqrt(std::abs(trans.linear().determinant()));
    const float sqr_threshold = square(threshold * std::min(scale, 1.f));

    return (mat(0,2) + mat(0,0) * point_pairs[0] + mat(0,1) * point_pairs[1] - point_pairs[2]).square()
       +(mat(1,2) + mat(1,0) * point_pairs[0] + mat(1,1) * point_pairs[1] - point_pairs[3]).square() < sqr_threshold;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkProjectiveTransformation(
    const Eigen::Projective2f& trans, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold)
{
    const auto& mat = trans.matrix();

    const Eigen::ArrayXf z = (mat(2, 2) + mat(2, 0) * point_pairs[0] + mat(2, 1) * point_pairs[1]);

    const float sqr_threshold = square(threshold);
    const Eigen::ArrayXf sqr_scale = z.square().min(std::abs(trans.linear().determinant()));

    return ((mat(0,2) + mat(0,0) * point_pairs[0] + mat(0,1) * point_pairs[1]) - point_pairs[2] * z).square()
        + ((mat(1,2) + mat(1,0) * point_pairs[0] + mat(1,1) * point_pairs[1]) - point_pairs[3] * z).square() < sqr_threshold * sqr_scale;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkEpipolarTransformation(
    const Eigen::Matrix3f& F, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold)
{
    const auto& x0 = point_pairs[0];
    const auto& y0 = point_pairs[1];
    const auto& x1 = point_pairs[2];
    const auto& y1 = point_pairs[3];

    //@todo: do one of the following:
    // 1. if number of points is high, use a for loop with each iteration calculating ~1024 points (fit in 64K L1 cache per core).
    // 2. optimise by using point_pairs as padded Eigen::ArrayX4f. check Eigen to see if that results in aligned load
    const Eigen::ArrayXf epipole0[3] = {
        x1 * F(0, 0) + (y1 * F(1, 0) + F(2, 0)),
        x1 * F(0, 1) + (y1 * F(1, 1) + F(2, 1)),
        x1 * F(0, 2) + (y1 * F(1, 2) + F(2, 2))
    };
    const Eigen::ArrayXf epipole1[2] = {
        x0 * F(0, 0) + (y0 * F(0, 1) + F(0, 2)),
        x0 * F(1, 0) + (y0 * F(1, 1) + F(1, 2))//,
        //x0 * F(2, 0) + (y0 * F(2, 1) + F(2, 2)) //not used
    };

    //(X1' * F * X0).squared()
    const Eigen::ArrayXf squared_X1_FX0 = (epipole0[0] * x0 + (epipole0[1] * y0 + epipole0[2])).square();

    const float squared_threshold = threshold * threshold;

    Eigen::Array<bool, Eigen::Dynamic, 1> mask = (squared_X1_FX0 < squared_threshold * (epipole0[0].square() + epipole0[1].square()))
        && (squared_X1_FX0 < squared_threshold * (epipole1[0].square() + epipole1[1].square()));

    return mask;
}

Eigen::Isometry3f solvePnP(const Eigen::Array<float, 4, 3>& pts3d, const Eigen::Array<float, 4, 2>& pts2d, const PinHoleCamera& cam)
{
    const InversePinHoleCamera camInv = cam.inverse();
    // use thread_local static to avoid repeated malloc/free
    thread_local std::vector<Eigen::Vector3d> x3d(4);
    thread_local std::vector<Eigen::Vector3d> x2d(4);
    thread_local std::vector<lambdatwist::CameraPose> poseCandidates(4);
    
    for (uint32_t i = 0; i < 4u; i++) {
        x3d[i] = Eigen::Vector3d{pts3d(i, 0), pts3d(i, 1), pts3d(i, 2)};
        x2d[i] = Eigen::Vector3d{
            double{pts2d(i, 0) * camInv.f.x + camInv.c.x},
            double{pts2d(i, 1) * camInv.f.y + camInv.c.y},
            1.0}.normalized();
    }
    poseCandidates.clear();
    int const nbPoseCandidates = lambdatwist::p3p(x2d, x3d, &poseCandidates);
    if (nbPoseCandidates == 0) {
        return Eigen::Isometry3f::Identity();
    }
    int idxBest = 0;
    double minErr = INFINITY;
    for (int i = 0; i < nbPoseCandidates; i++) {
        auto const& pose = poseCandidates.at(i);
        auto const err = ((pose.R * x3d.at(3) + pose.t).hnormalized() - x2d.at(3).hnormalized()).squaredNorm();
        if (err < minErr) {
            idxBest = i;
            minErr = err;
        }
    }
    auto const& pose = poseCandidates.at(idxBest);
    const auto R = pose.R;
    const auto t = pose.t;
    Eigen::Isometry3f trans = Eigen::Isometry3f::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            trans.linear()(i, j) = R(i, j);
        }
        trans.translation()[i] = t[i];
    }
    return trans;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkPnP(const Eigen::Isometry3f& trans, const std::array<Eigen::ArrayXf, 3>& pts3d,
    const std::array<Eigen::ArrayXf, 2>& pts2d, const PinHoleCamera& cam, const float threshold)
{
    const Eigen::Matrix<float, 3, 4> P = toKMat<float>(cam) * trans.matrix().template topRows<3>();
#define RSFM_PROJ(idx) (P(idx, 3) + P(idx, 0) * pts3d[0] + P(idx, 1) * pts3d[1] + P(idx, 2) * pts3d[2])
    //@fixme: check which is faster. If compiler common expression elimination works well with Eigen, the second should be faster.
#if 0
    const Eigen::ArrayXf zInv = RSFM_PROJ(2).inverse();
    return (RSFM_PROJ(0) * zInv - pts2d[0]).square() + (RSFM_PROJ(1) * zInv - pts2d[1]).square() < square(threshold);
#else
    return (RSFM_PROJ(0) * RSFM_PROJ(2).inverse() - pts2d[0]).square() + (RSFM_PROJ(1) * RSFM_PROJ(2).inverse() - pts2d[1]).square() < square(threshold);
#endif
#undef RSFM_PROJ
}

Eigen::MatrixX3f triangulate(
        const PinHoleCamera &camera0, const PinHoleCamera &camera1, const Eigen::Isometry3f &RT,
        const std::array<Eigen::ArrayXf, 4> &pairs)
{
    const int nbPairs = pairs.at(0).rows();    
    Eigen::MatrixX3f pts3d(nbPairs, 3);

    const InversePinHoleCamera camInv0 = camera0.inverse();
    const InversePinHoleCamera camInv1 = camera1.inverse();

    using UnalignedVec3f = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;
    using UnalignedMat3f = Eigen::Matrix<float, 3, 3, Eigen::DontAlign>;

    const Eigen::Isometry3f invRT = RT.inverse();
    const UnalignedMat3f invR1 = invRT.linear();
    const UnalignedVec3f C1 = invRT.translation();

    const auto toRay = [](const InversePinHoleCamera& camInv, float x, float y){
        return (UnalignedVec3f{} << camInv.f.x * x + camInv.c.x, camInv.f.y * y + camInv.c.y, 1).finished();
    };

    VECTORIZE_LOOP
    for(int i = 0; i < nbPairs; i++)
    {
        const float x0 = pairs[0][i], y0 = pairs[1][i], x1 = pairs[2][i], y1 = pairs[3][i];
        const UnalignedVec3f ray0 = toRay(camInv0, x0, y0).normalized();
        const UnalignedVec3f ray1 = (invR1 * toRay(camInv1, x1, y1)).normalized();
        const UnalignedMat3f M0 = ray0 * ray0.transpose() - UnalignedMat3f::Identity();
        const UnalignedMat3f M1 = ray1 * ray1.transpose() - UnalignedMat3f::Identity();

        const UnalignedMat3f A = M0 + M1;
        const UnalignedVec3f b = M1 * C1; // C0 is Zero()
#if 0
        // gauss elimination
        Eigen::Matrix<float, 3, 4, Eigen::DontAlign> Ab;
        Ab << A, b;
        for (int i = 0; i < A.rows(); i++) {
            const float inv = 1.f / Ab(i, i);
            for (int j = i + 1; j < A.rows(); j++) {
                const float factor = Ab(j, i) * inv;
                for (int k = i + 1; k < A.cols() + b.cols(); k++) {
                    Ab(j, k) -= factor * Ab(i, k);
                }
            }
        }
        UnalignedVec3f pt3d = Ab.template rightCols<1>();
        for (int i = A.rows() - 1; i >= 0; i--) {
            const float inv = 1.f / Ab(i, i);
            pt3d.row(i) *= inv;
            for (int j = i - 1; j >= 0; j--){
                const float factor = Ab(j, i);
                pt3d.row(j) -= factor * pt3d.row(i);
            }
        }
        pts3d.row(i) = pt3d.transpose();
#else
        const UnalignedVec3f pt3d = A.inverse() * b;
        pts3d.row(i) = pt3d.transpose();
#endif
    }
    return pts3d;
}

template <typename T>
Eigen::Matrix<T, 3, 3> crossProdMat(Eigen::Matrix<T, 3, 1> const& v)
{
    Eigen::Matrix<T, 3, 3> m;
    m << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    return m;
}

//! \returns number of solutions
// Ref: https://imkaywu.github.io/blog/2017/06/fundamental-matrix/
uint32_t findEpipolarityWith7Points(const Eigen::Matrix<float, 7, 4>& pts, Eigen::Matrix3f* solutions){
    Affine2f trans[2];
    Eigen::Matrix<float, 7, 4> pts_norm;
    std::tie(trans[0], trans[1], pts_norm) = hartleyNormalize(pts);
    Eigen::Matrix<float, 7, 9> P(pts.rows(), 9);

    //e0*x*x_ + e1*x*y_ + e2*x + e3*x_*y + e4*y*y_ + e5*y + e6*x_ + e7*y_ + e8, e in col-major 3x3 matrix
#define x0 pts_norm.array().col(0)
#define y0 pts_norm.array().col(1)
#define x1_y1 pts_norm.array().template rightCols<2>()
#define vec_ones Eigen::Matrix<float, 7, 1>::Ones(pts.rows())
    P << x1_y1.colwise() * x0, x0, x1_y1.colwise() * y0, y0, x1_y1, vec_ones;
#undef vec_ones
#undef x0
#undef y0
#undef x1_y1

    // solving for nullspace of A to get two F
    auto svdP = P.jacobiSvd(Eigen::ComputeFullV);

    const Eigen::Vector<float, 9> v1 = svdP.matrixV().col(7);
    const Eigen::Vector<float, 9> v2 = svdP.matrixV().col(8);

    const Eigen::Matrix3f F1F2[2] = {Eigen::Matrix3f::Map(v1.data()), Eigen::Matrix3f::Map(v2.data())};

    // find F that meets the singularity constraint: det(a * F1 + (1 - a) * F2) = 0
    float D[2][2][2];
    for (int i1 = 0; i1 < 2; ++i1) {
        for (int i2 = 0; i2 < 2; ++i2) {
            for (int i3 = 0; i3 < 2; ++i3) {
                Eigen::Matrix3f Dtmp;
                Dtmp.col(0) = F1F2[i1].col(0);
                Dtmp.col(1) = F1F2[i2].col(1);
                Dtmp.col(2) = F1F2[i3].col(2);
                D[i1][i2][i3] = Dtmp.determinant();
            }
        }
    }
    
    // solving cubic equation and getting 1 or 3 solutions for F
    Eigen::Vector4f coefficients;
    coefficients(0) = -D[1][0][0]+D[0][1][1]+D[0][0][0]+D[1][1][0]+D[1][0][1]-D[0][1][0]-D[0][0][1]-D[1][1][1];
    coefficients(1) = D[0][0][1]-2*D[0][1][1]-2*D[1][0][1]+D[1][0][0]-2*D[1][1][0]+D[0][1][0]+3*D[1][1][1];
    coefficients(2) = D[1][1][0]+D[0][1][1]+D[1][0][1]-3*D[1][1][1];
    coefficients(3) = D[1][1][1];

    coefficients *= 1.f / coefficients[3];

    Eigen::Matrix3f companionMat;
    companionMat << 0,0,-coefficients[0],
        1, 0, -coefficients[1],
        0, 1, -coefficients[2];
    Eigen::Vector3cf const roots = companionMat.eigenvalues();


    Eigen::Matrix3f st[2];
    for(int i = 0; i < 2; i++)
        st[i] << trans[i].matrix(),
                0, 0, 1;

    // check sign consistency
    uint32_t nbSolutions = 0;
    for (int i = 0; i < roots.rows(); i++)
    {
        if (roots[i].imag() != 0) {
            continue;
        }
        float const a = roots[i].real();
        Eigen::Matrix3f const F = a * F1F2[0] + (1-a) * F1F2[1];
#if 0
        auto const svdFt = F.transpose().jacobiSvd(Eigen::ComputeFullV);
        Eigen::Vector3f const e1 = svdFt.matrixV().col(2);
        Eigen::Matrix<float, 3, 7> const l1_ex = crossProdMat(e1) * pts_norm.leftCols<2>().transpose().colwise().homogeneous(); // lines connecting of x1 and e1
        Eigen::Matrix<float, 3, 7> const l1_Fx = F * pts_norm.rightCols<2>().transpose().colwise().homogeneous();    // lines determined by F and x2
        auto const s = (l1_Fx.array() * l1_ex.array()).colwise().sum().eval();
        if (!(s > 0).all() && !(s < 0).all())
        {
            printf("[Debug] Sign consistency check failed for 7-point algorithm\n");
            continue;
        }
#endif
        solutions[nbSolutions++] = st[1].transpose() * F * st[0];
    }

    HOPE(nbSolutions > 0 && nbSolutions <= 3);
    return nbSolutions;
}

// @fixme: Not sure if this is correct. Need testing
// Ref: Two-view Geometry Estimation Unaffected by a Dominant Plane
Eigen::Projective2f findHomographyFromEpipolarity(Eigen::Matrix3f const& F, Eigen::Matrix<float, 3, 4> const& pts)
{
    Eigen::Vector3d const e_ = F.cast<double>().transpose().jacobiSvd(Eigen::ComputeFullV).matrixV().col(2); // need transpose?
    Eigen::Matrix3d const A = crossProdMat(e_) * F.cast<double>();
    Eigen::Vector3d b;
    for (int i = 0; i < b.rows(); i++)
    {
        Eigen::Vector3d const x = pts.template block<1, 2>(i, 0).transpose().homogeneous().cast<double>();
        Eigen::Vector3d const x_ = pts.template block<1, 2>(i, 2).transpose().homogeneous().cast<double>();
        Eigen::Vector3d const tmp = x_.cross(e_);
        b[i] = x_.cross(A*x).dot(tmp) / tmp.squaredNorm();
    }
    Eigen::Matrix3d M = pts.template leftCols<2>().rowwise().homogeneous().cast<double>();
    return Eigen::Projective2f{(A - e_ * (M.householderQr().solve(b)).transpose()).eval().cast<float>()};
}

// @fixme: Not sure if this is correct. Need testing
// Ref: https://math.stackexchange.com/a/1973389
//      page 336, Multiple View Geometry in Computer Vision (2nd edition)
Eigen::Matrix3f findEpipolarityFromHomography(Eigen::Projective2f const& H, Eigen::Matrix<float, 2, 4> const& pts)
{
    Eigen::Vector3f lines[2];
    for (int i = 0; i < 2; i++) {
        Eigen::Vector3f const x = pts.template block<1, 2>(i, 0).transpose().homogeneous();
        Eigen::Vector3f const x_ = pts.template block<1, 2>(i, 2).transpose().homogeneous();
        lines[i] = (H.matrix() * x).cross(x_);
    }
    return crossProdMat(lines[0].cross(lines[1])) * H.matrix();
}

Eigen::Matrix<float, Eigen::Dynamic, 4> mask2pts(const std::array<Eigen::ArrayXf, 4>& ptPairs, const bool* mask) {
    const long nbValid = std::count(mask, mask + ptPairs[0].rows(), true);
    Eigen::Matrix<float, Eigen::Dynamic, 4> pts(nbValid, 4);
    for(int idx = 0, i = 0; i < ptPairs[0].rows(); i++)
    {
        if (mask[i])
        {
            assert(idx < nbValid);
            pts.row(idx) <<  ptPairs[0][idx], ptPairs[1][idx], ptPairs[2][idx], ptPairs[3][idx];
            idx++;
        }
    }
    return pts;
}

} // namespace rsfm
