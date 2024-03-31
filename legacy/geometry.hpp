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

#pragma once
#include <eigen3/Eigen/Eigen>
#include "../Types.hpp"
#include "../SfmUtils.hpp"
#include <macros.h>
#include "../RapidSFM.h"
#include <numeric>
namespace rsfm
{

// For matrix representation of cross product. cross_vec2mat(a) * b == cross(a, b);
template<typename T>
Matrix3<T> cross_vec2mat(const Vector3<T>& vec)
{
    Matrix3<T> mat;
    mat <<  0, -vec(2), vec(1),
            vec(2), 0, -vec(0),
            -vec(1), vec(0), 0;
    return mat;
}

template <typename T>
Vector3<T> cross_mat2vec(const Matrix3<T>& mat)
{
    Vector3<T> vec;
    vec <<  mat(2, 1) - mat(1, 2),
            mat(0, 2) - mat(2, 0),
            mat(1, 0) - mat(0, 1);
    return T(0.5f) * vec;
}

template<typename Derived>
Isometry2f findSim2(const Eigen::MatrixBase<Derived>& pts)
{
    Eigen::Matrix<float, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, Derived::ColsAtCompileTime> A(pts.rows() * 2, 4);
    Eigen::Matrix<float, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, 1> b(pts.rows() * 2, 1);

    //@todo: use vector operations to fill A and b instead of for loops
    for(int i = 0; i < pts.rows(); i++)
    {
        A.template block<2, 4>(i * 2, 0)
                << pts(i, 0), -pts(i, 1), 1, 0,
                pts(i, 1), pts(i, 0), 0, 1;
        b.template block<2, 1>(i * 2, 0)
                << pts(i, 2), pts(i, 3);
    }

    Eigen::Vector4f X = least_square_solve(A, b);

//    A <<    pts(0, 0), -pts(0, 1), 1, 0,
//            pts(0, 1), pts(0, 0), 0, 1,
//            pts(1, 0), -pts(1, 1), 1, 0,
//            pts(1, 1), pts(1, 0), 0, 1;
//    b <<    pts(0, 2), pts(0, 3),
//            pts(1, 2), pts(1, 3);
    //Eigen::Vector4d X = A.colPivHouseholderQr().solve(b);

    Isometry2f result = Isometry2f::Identity();
    result.matrix().topRows<2>() << X[0], -X[1], X[2],
            X[1], X[0], X[3];

    return result;
}

template<typename Derived>
Affine2f findAffine2(const Eigen::MatrixBase<Derived>& pts)
{
    Eigen::Matrix<float, Derived::RowsAtCompileTime, 3> A = pts.template leftCols<2>().rowwise().homogeneous();
    Eigen::Matrix<float, Derived::RowsAtCompileTime, 2> b = pts.template rightCols<2>();

    Affine2f result;

    if(pts.rows() == 3) {
        result.matrix().transpose() = A.inverse() * b;
    }
    else
        result.matrix().transpose() = leastSquareSolve(A, b);

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkAffine2Transformation(
        const Affine2f &trans, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold);

template<typename Derived>
std::tuple<Affine2<typename Derived::Scalar>, Affine2<typename Derived::Scalar>, Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    hartleyNormalize(const Eigen::MatrixBase<Derived>& pts)
{
    ASSERT(pts.cols() == 4 && "should have 4 columns: ax, ay, bx, by");
    typedef typename Derived::Scalar Scalar;
    Eigen::Matrix<Scalar, 2, 4> bound;
    bound.row(0) = pts.colwise().minCoeff();
    bound.row(1) = pts.colwise().maxCoeff();
    Eigen::Matrix<Scalar, 1, 4> centre = pts.colwise().mean();
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm(pts.rows(), pts.cols());
    pts_norm << pts.template leftCols<2>().rowwise() - centre.template leftCols<2>(), pts.template rightCols<2>().rowwise() - centre.template rightCols<2>();

    const Scalar scale[2] = {
            std::sqrt(2.f) / pts_norm.template leftCols<2>().rowwise().norm().mean(),
            std::sqrt(2.f) / pts_norm.template rightCols<2>().rowwise().norm().mean()
    };

    Affine2<Scalar> trans[2];
    trans[0].matrix() << scale[0], 0, scale[0] * -centre[0],
            0, scale[0], scale[0] * -centre[1];
    trans[1].matrix() << scale[1], 0, scale[1] * -centre[2],
            0, scale[1], scale[1] * -centre[3];


    pts_norm.template leftCols<2>() *= scale[0];
    pts_norm.template rightCols<2>() *= scale[1];

    return std::make_tuple(trans[0], trans[1], pts_norm);
}

//based on https://math.stackexchange.com/a/3511513
//an alternative method is here: https://towardsdatascience.com/estimating-a-homography-matrix-522c70ec4b2c
template<typename Derived>
Eigen::Transform<typename Derived::Scalar, 2, Eigen::Projective> findHomography(
        const Eigen::MatrixBase<Derived>& pts, bool preferAccuracy = true)
{
    ASSERT(pts.rows() >= 4);
    typedef typename Derived::Scalar Scalar;
    Affine2<Scalar> trans[2];
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm;
    std::tie(trans[0], trans[1], pts_norm) = hartleyNormalize(pts);
    //trans[0].setIdentity(); trans[1].setIdentity(); pts_norm = pts;
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, 9> P(pts.rows() * 2, 9);

//    auto x = pts_norm.col(0).array().eval();
//    auto y = pts_norm.col(1).array().eval();
//    auto x_ = pts_norm.col(2).array().eval();
//    auto y_ = pts_norm.col(3).array().eval();
//    typedef decltype(x) vec_type;
//    P << -x, -y, -vec_type::Ones(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), x*x_, y*x_, x_,
//            vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), -x, -y, -vec_type::Ones(), x*y_, y*y_, y_;

#define x0_y0 pts_norm.array().template leftCols<2>()
#define x1 pts_norm.array().col(2)
#define y1 pts_norm.array().col(3)
#define vec_ones Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 1>::Ones(pts.rows())
    P << -x0_y0, -vec_ones, Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 3>::Zero(pts.rows(), 3), x0_y0.colwise() * x1, x1,
        Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 3>::Zero(pts.rows(), 3), -x0_y0, -vec_ones, x0_y0.colwise() * y1, y1;
#undef vec_ones
#undef x1
#undef y1
#undef x0_y0
//    //@todo: use vector operations to fill P instead of for loops - done
//    for(int i = 0; i < pts.rows(); i++)
//    {
//        const Scalar x = pts_norm(i, 0);
//        const Scalar y = pts_norm(i, 1);
//        const Scalar x_ = pts_norm(i, 2);
//        const Scalar y_ = pts_norm(i, 3);
//        P.template middleRows<2>(i * 2)
//                << -x, -y, -1.f, 0.f, 0.f, 0.f, x*x_, y*x_, x_,
//                0.f, 0.f, 0.f, -x, -y, -1, x*y_, y*y_, y_;
//    }

    Eigen::Matrix<Scalar, 9, 1> h_norm;
    constexpr bool warEigenSegFault = EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 4 && EIGEN_MINOR_VERSION == 0;
    if(preferAccuracy && !warEigenSegFault){
        //more accurate
        auto svd = P.jacobiSvd(Eigen::ComputeFullV); // segfaults with eigen 3.4.0-r1
        h_norm = svd.matrixV().col(8);
    }
    else {
        //less accurate, but faster
        auto svd = (P.transpose() * P).eval().jacobiSvd(Eigen::ComputeFullV);
        h_norm = svd.matrixV().col(8);
    }

    Eigen::Matrix<Scalar, 3, 3> st[2];
    for(int i = 0; i < 2; i++)
        st[i] << trans[i].matrix(),
                0, 0, 1;
    Eigen::Transform<Scalar, 2, Eigen::Projective> H;
    H.matrix() = st[1].colPivHouseholderQr().solve(Eigen::Matrix<Scalar, 3,3, Eigen::RowMajor>::Map(h_norm.data())) * st[0];
    return H;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkProjectiveTransformation(
        const Eigen::Projective2f& trans, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold);

//! \returns number of solutions
// Ref: https://imkaywu.github.io/blog/2017/06/fundamental-matrix/
uint32_t findEpipolarityWith7Points(const Eigen::Matrix<float, 7, 4>& pts, Eigen::Matrix3f* solutions);

//! Find homography matrix given fundamental matrix and three point correspondence
// See Degensac papar
Eigen::Projective2f findHomographyFromEpipolarity(Eigen::Matrix3f const& F, Eigen::Matrix<float, 3, 4> const& pts);

Eigen::Matrix3f findEpipolarityFromHomography(Eigen::Projective2f const& H, Eigen::Matrix<float, 2, 4> const& pts);

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> findEpipolarity(const Eigen::MatrixBase<Derived>& pts, bool preferAccuracy = true){
    ASSERT(pts.rows() >= 8 && "at least 8 points are required");
    typedef typename Derived::Scalar Scalar;
    Affine2<Scalar> trans[2];
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm;
    std::tie(trans[0], trans[1], pts_norm) = hartleyNormalize(pts);
    Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 9> P(pts.rows(), 9);

    //e0*x*x_ + e1*x*y_ + e2*x + e3*x_*y + e4*y*y_ + e5*y + e6*x_ + e7*y_ + e8, e in col-major 3x3 matrix
#define x0 pts_norm.array().col(0)
#define y0 pts_norm.array().col(1)
#define x1_y1 pts_norm.array().template rightCols<2>()
#define vec_ones Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, 1>::Ones(pts.rows())
    P << x1_y1.colwise() * x0, x0, x1_y1.colwise() * y0, y0, x1_y1, vec_ones;
#undef vec_ones
#undef x0
#undef y0
#undef x1_y1

    Eigen::Matrix<Scalar, 9, 1> f_norm;
    if(preferAccuracy){
        //more accurate
        auto svd = P.jacobiSvd(Eigen::ComputeFullV);
        f_norm = svd.matrixV().col(8);
    }else {
        //less accurate, but faster
        auto svd = (P.transpose() * P).eval().jacobiSvd(Eigen::ComputeFullV);
        f_norm = svd.matrixV().col(8);
    }

    Eigen::Matrix<Scalar, 3, 3> st[2];
    for(int i = 0; i < 2; i++)
        st[i] << trans[i].matrix(),
                0, 0, 1;
    Eigen::Matrix<Scalar, 3, 3> F_est = Eigen::Matrix<Scalar, 3,3>::Map(f_norm.data());

    auto svd_f33 = F_est.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<Scalar, 3, 1> S = svd_f33.singularValues();
    S[2] = 0;
    Eigen::Matrix<Scalar, 3, 3> F = st[1].transpose() * svd_f33.matrixU() * S.asDiagonal() * svd_f33.matrixV().transpose() * st[0];

    return F;
}

Eigen::Array<bool, Eigen::Dynamic, 1> checkEpipolarTransformation(
        const Eigen::Matrix3f& F, const std::array<Eigen::ArrayXf, 4>& point_pairs, const float threshold);

Eigen::Isometry3f solvePnP(const Eigen::Array<float, 4, 3>& pts3d, const Eigen::Array<float, 4, 2>& pts2d, const PinHoleCamera& cam);

Eigen::Array<bool, Eigen::Dynamic, 1> checkPnP(const Eigen::Isometry3f& trans, const std::array<Eigen::ArrayXf, 3>& pts3d,
    const std::array<Eigen::ArrayXf, 2>& pts2d, const PinHoleCamera& cam, const float threshold);

template <typename Scalar>
std::array<Isometry3<Scalar>, 4> decomposeEssentialMatrix(const Matrix3<Scalar>& E)
{
    /*https://en.wikipedia.org/wiki/Essential_matrix*/
    Eigen::JacobiSVD<Matrix3<Scalar>> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3<Scalar> U = svd.matrixU();
    Matrix3<Scalar> V = svd.matrixV();
    // Vector3<Scalar> S = svd.singularValues();
    if(U.determinant() < 0)
    {
        U.col(2) = -U.col(2);
        // S[2] = -S[2]; // S[2] is always 0
    }
    if(V.determinant() < 0)
    {
        V.col(2) = -V.col(2);
        // S[2] = -S[2];
    }
    // printf("S = [%f, %f, %f]\n", (float)S[0], (float)S[1], (float)S[2]);

    Matrix3<Scalar> W;
    W <<    0, -1, 0,
            1, 0, 0,
            0, 0, 1;
    Matrix3<Scalar> R[2] = {
            U * W.transpose() * V.transpose(),
            U * W * V.transpose()
    };
    assert(R[0].determinant() > 0 && R[1].determinant() > 0);

    std::array<Isometry3<Scalar>, 4> solution_list;
    Vector3<Scalar> T[2];
    // @fixme: add tests
#if 0
    Matrix3<Scalar> Z;
    Z <<    0, 1, 0,
            -1, 0, 0,
            0, 0, 0;
    //const Matrix3<Scalar> Tx = U * W * S.asDiagonal() * U.transpose();
    const Matrix3<Scalar> Tx = U * Z * U.transpose();
    T[0] << cross_mat2vec(Tx).normalized();
    T[1] = -T[0];

    // printf("T = [%f, %f, %f] / [%f, %f, %f]\n", (float)T[0][0], (float)T[0][1], (float)T[0][2], (float)U(0, 2),(float)U(1, 2),(float)U(2, 2));

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            auto& solution = solution_list[i*2 + j];
            solution.setIdentity();
            solution.linear() = R[i];
            solution.translation() = T[j];
        }
    }
#else
    T[0] = U.col(2);
    T[1] = -T[0];

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            auto& solution = solution_list[i*2 + j];
            solution.setIdentity();
            solution.linear() = R[i];
            solution.translation() = T[j];
        }
    }
#endif
    return solution_list;
};

template <typename Scalar>
std::array<Isometry3<Scalar>, 4> decomposeEpipolarity(
        const PinHoleCamera& cam0, const PinHoleCamera& cam1,
        const Matrix3<Scalar>& F)
{
    /*https://en.wikipedia.org/wiki/Essential_matrix*/
    const Matrix3<Scalar> E = toKMat<Scalar>(cam1).transpose() * F * toKMat<Scalar>(cam0);
    return decomposeEssentialMatrix(E);
};

// decomposed homography
template<typename T>
struct HomographyFactors{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix3<T> R;
    Vector3<T> N;
    Vector3<T> T_by_d;// 1/d * T. T is translation and d is distance to homography plane

    //T is normalized in returned RT
    Isometry3<T> get_RT() const{
        Isometry3<T> RT(R);
        RT.translation() = T_by_d.normalized();
        return RT;
    }
};

//decompose_H_F.pdf
template<typename Scalar>
EigenAlignedVector<HomographyFactors<Scalar>> decomposeHomography(const Matrix3<Scalar>& H)
{
    //normalise
    Scalar scale = H.jacobiSvd().singularValues()[1];
    Matrix3<Scalar> H_norm = 1 / scale * H;

    auto svd = (H_norm.transpose() * H_norm).eval().jacobiSvd(Eigen::ComputeFullV);
    const Vector3<Scalar>& S = svd.singularValues();
    const Matrix3<Scalar> V = svd.matrixV().determinant() > 0 ? svd.matrixV() : (-svd.matrixV()).eval();
    assert(V.determinant() > 0);
    const Vector3<Scalar> v1 = V.col(0);
    const Vector3<Scalar> v2 = V.col(1);
    const Vector3<Scalar> v3 = V.col(2);

    Scalar c = 1 / std::sqrt(S[0] - S[2]);
    Scalar a = std::sqrt(1 - S[2]) * c;
    Scalar b = std::sqrt(S[0] - 1) * c;

    const Vector3<Scalar> u1 = a * v1 + b * v3;
    const Vector3<Scalar> u2 = a * v1 - b * v3;

    Matrix3<Scalar> U1, U2, W1, W2;
    U1 << v2, u1, v2.cross(u1);
    U2 << v2, u2, v2.cross(u2);

    std::array<HomographyFactors<Scalar>, 8> solution_list;
    for(int i = 0; i < 2; i++)
    {
        if(i == 1)
            H_norm = -H_norm;//negative of homography matrix is the same transformation

        W1 << H_norm * v2, H_norm * u1, (H_norm * v2).cross(H_norm * u1);
        W2 << H_norm * v2, H_norm * u2, (H_norm * v2).cross(H_norm * u2);
        {
            Matrix3<Scalar> R = W1 * U1.transpose();
            Vector3<Scalar> N = v2.cross(u1);
            Vector3<Scalar> T_by_d = (H_norm - R) * N;
            solution_list[i*4] = {R, N, T_by_d};
        }
        {
            Matrix3<Scalar> R = W2 * U2.transpose();
            Vector3<Scalar> N = v2.cross(u2);
            Vector3<Scalar> T_by_d = (H_norm - R) * N;
            solution_list[i*4 + 1] = {R, N, T_by_d};
        }
        {
            Matrix3<Scalar> R = solution_list[i*4].R;
            Vector3<Scalar> N = -solution_list[i*4].N;
            Vector3<Scalar> T_by_d = -solution_list[i*4].T_by_d;
            solution_list[i*4 + 2] = {R, N, T_by_d};
        }
        {
            Matrix3<Scalar> R = solution_list[i*4 + 1].R;
            Vector3<Scalar> N = -solution_list[i*4 + 1].N;
            Vector3<Scalar> T_by_d = -solution_list[i*4 + 1].T_by_d;
            solution_list[i*4 + 3] = {R, N, T_by_d};
        }
    }
    std::array<bool, 8> mask;
    for(int i = 0; i < 8; i++) {
        mask[i] = solution_list[i].N.dot(solution_list[i].T_by_d) < 1;
    }

    EigenAlignedVector<HomographyFactors<Scalar>> valid_solutions;
    valid_solutions.reserve(std::count(mask.begin(), mask.end(), true));
    for(unsigned i = 0; i < solution_list.size(); i++)
        if(mask[i])
            valid_solutions.emplace_back(solution_list[i]);

    return valid_solutions;
};

template<typename Scalar>
EigenAlignedVector<HomographyFactors<Scalar>> decomposeHomography(
        const PinHoleCamera& cam0, const PinHoleCamera& cam1,
        const Matrix3<Scalar>& G
)
{
    //normalise
    Matrix3<Scalar> H = toKMat<Scalar>(cam1).inverse() * G * toKMat<Scalar>(cam0);
    return decomposeHomography(H);
};

#if 0
#include <opencv2/calib3d.hpp>
template<typename Scalar>
std::vector<homography_decomposed_t<Scalar>> decomposeHomographyOpencv(
        const Camera& cam0, const Camera& cam1,
        const Matrix3<Scalar>& G
)
{
    unused(cam1);
    std::vector<cv::Mat> R_list, T_list, N_list;
    cv::Mat H(3, 3, CV_32F);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            H.at<float>(i,j) = G(i,j);
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0,0) = cam0.f.x;
    K.at<float>(1,1) = cam0.f.y;
    K.at<float>(0,2) = cam0.c.x;
    K.at<float>(1,2) = cam0.c.y;
    cv::decomposeHomographyMat(H, K, R_list, T_list, N_list);
    std::vector<homography_decomposed_t<Scalar>> result;
    for(unsigned n = 0; n < R_list.size(); n++){
        homography_decomposed_t<Scalar> solution;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                solution.R(i,j) = R_list[n].at<float>(i,j);
        for(int i = 0; i < 3; i++)
            solution.N[i] = N_list[n].at<float>(i);
        for(int i = 0; i < 3; i++)
            solution.T_by_d[i] = T_list[n].at<float>(i);
        result.push_back(solution);
    }
    return result;
};
#endif

//triangulate a single point from multiple lines
//@todo: add unit test
//https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
template <size_t nbObsStatic = kInvalid<size_t>>
Vec3<float> triangulate(const std::tuple<const Pose*, const PinHoleCamera*, Vec2<float>>* observList, size_t nbObs = kInvalid<size_t>){
    if constexpr (nbObsStatic != kInvalid<size_t>) {
        assert(nbObs == kInvalid<size_t>);
        nbObs = nbObsStatic;
    }
    Matrix3<float> A = Matrix3<float>::Zero();
    Vector3<float> b = Vector3<float>::Zero();
    for(unsigned i = 0; i < nbObs; i++){
        const auto& ob = observList[i];
        const Pose& pose = *std::get<0>(ob);
        const PinHoleCamera& camera = *std::get<1>(ob);
        const Vec2<float>& pt2d = std::get<2>(ob);

        const Vector3<float> p = toEigen(pose.C);

        Vector3<float> v;
        v << (pt2d.x - camera.c.x) / camera.f.x, (pt2d.y - camera.c.y) / camera.f.y, 1.f;
        v = toEigen(pose.R).conjugate() * v;
        v.normalize();

        const Matrix3<float> M = Matrix3<float>::Identity() - v * v.transpose();
        A += M;
        b += M * p;
    }
    return fromEigen((A.inverse() * b).eval());
}

// For this one, std::get<1>(observList[x]) is the uv coordinates
template <size_t nbObsStatic = kInvalid<size_t>>
Vec3<float> triangulate(const std::pair<const Pose*, Vec2<float>>* observList, size_t nbObs = kInvalid<size_t>){
    if constexpr (nbObsStatic != kInvalid<size_t>) {
        assert(nbObs == kInvalid<size_t> || nbObs == nbObsStatic);
        nbObs = nbObsStatic;
    }
    Matrix3<float> A = Matrix3<float>::Zero();
    Vector3<float> b = Vector3<float>::Zero();
    for(unsigned i = 0; i < nbObs; i++){
        const auto& ob = observList[i];
        const Pose& pose = *std::get<0>(ob);
        const Vec2<float>& uv = std::get<1>(ob);

        const Vector3<float> p = toEigen(pose.C);

        Vector3<float> v;
        v << uv.x, uv.y, 1.f;
        v = toEigen(pose.R).conjugate() * v;
        v.normalize();

        const Matrix3<float> M = Matrix3<float>::Identity() - v * v.transpose();
        A += M;
        b += M * p;
    }
    return fromEigen((A.inverse() * b).eval());
}

template<typename T, typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 3> triangulate(
        const PinHoleCamera &camera0, const PinHoleCamera &camera1, const Isometry3<T> &RT,
        const Eigen::MatrixBase<Derived> &pairs);// __attribute__((optimize ("fast-math")));

//@todo: add unit test
template<typename T, typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 3> triangulate(
        const PinHoleCamera &camera0, const PinHoleCamera &camera1, const Isometry3<T> &RT,
        const Eigen::MatrixBase<Derived> &pairs)
{
    typedef typename Derived::Scalar Scalar;
    assert(pairs.cols() == 4);
    //use Eigen::Array as scalar type of to solve all vertex pairs in a batch?
    const std::array<Eigen::Array<Scalar, Derived::RowsAtCompileTime, 1>, 4> pairCols{{
            pairs.col(0), pairs.col(1),
            pairs.col(2), pairs.col(3)
    }};
    return triangulate(camera0, camera1, RT, pairCols);
}

#if 1
template <typename Scalar, int rowsAtCompileTime>
Eigen::Matrix<Scalar, rowsAtCompileTime, 3> triangulate(
        const PinHoleCamera &camera0, const PinHoleCamera &camera1, const Isometry3<Scalar> &RT,
        const std::array<Eigen::Array<Scalar, rowsAtCompileTime, 1>, 4> &pairs)
{
    //just use Eigen::Array as scalar type of to solve all vertex pairs in a batch?
    const Eigen::Array<Scalar, rowsAtCompileTime, 1>&
            X0 = pairs.at(0), Y0 = pairs.at(1),
            X1 = pairs.at(2), Y1 = pairs.at(3);
    ASSERT(allEqual(X0.rows(), Y0.rows(), X1.rows(), Y1.rows()));
    const int num_pairs = X0.rows();

    Eigen::Matrix<Scalar, rowsAtCompileTime, 1> vec_x(num_pairs, 1), vec_y(num_pairs, 1), vec_z(num_pairs, 1);
    Eigen::Matrix<Scalar, 3, 4> P = (toKMat<Scalar>(camera1)*RT.matrix().template topRows<3>()).template cast<Scalar>();
    const Scalar P00 = P(0, 0), P01 = P(0, 1), P02 = P(0, 2), P03 = P(0, 3), P10 = P(1, 0), P11 = P(1, 1), P12 = P(1, 2), P13 = P(1, 3), P20 = P(2, 0), P21 = P(2, 1), P22 = P(2, 2), P23 = P(2, 3);
    const Scalar fx0 = camera0.f.x, fy0 = camera0.f.y, cx0 = camera0.c.x, cy0 = camera0.c.y;
    //factors for A[3][3]
    const Scalar CA00[5] = { P20*P20, P00*P00 + P10*P10 + fx0*fx0, -2*P00*P20, -2*P10*P20, P20*P20};
    const Scalar CA01[5] = { P20*P21, P00*P01 + P10*P11, -P00*P21 - P01*P20, -P10*P21 - P11*P20, P20*P21};
    const Scalar CA02[6] = { P00*P02 + P10*P12 + cx0*fx0, -P00*P22 - P02*P20, P20*P22, -P10*P22 - P12*P20, P20*P22, -fx0};
    const Scalar CA10[5] = { P20*P21, P00*P01 + P10*P11, -P00*P21 - P01*P20, -P10*P21 - P11*P20, P20*P21};
    const Scalar CA11[5] = { P21*P21, P01*P01 + P11*P11 + fy0*fy0, -2*P01*P21, -2*P11*P21, P21*P21};
    const Scalar CA12[6] = { P01*P02 + P11*P12 + cy0*fy0, -P01*P22 - P02*P21, -fy0, P21*P22, -P11*P22 - P12*P21, P21*P22};
    const Scalar CA20[6] = { P00*P02 + P10*P12 + cx0*fx0, -P00*P22 - P02*P20, P20*P22, -P10*P22 - P12*P20, P20*P22, -fx0};
    const Scalar CA21[6] = { P01*P02 + P11*P12 + cy0*fy0, -P01*P22 - P02*P21, -fy0, P21*P22, -P11*P22 - P12*P21, P21*P22};
    const Scalar CA22[9] = { P02*P02 + P12*P12 + cx0*cx0 + cy0*cy0, -2*P02*P22, 1, P22*P22, 1, -2*P12*P22, -2*cx0, P22*P22, -2*cy0};
    const Scalar Cb0[5] = { -P20*P23, -P00*P03 - P10*P13, P00*P23 + P03*P20, P10*P23 + P13*P20, -P20*P23};
    const Scalar Cb1[5] = { -P21*P23, -P01*P03 - P11*P13, P01*P23 + P03*P21, P11*P23 + P13*P21, -P21*P23};
    const Scalar Cb2[5] = { -P22*P23, -P02*P03 - P12*P13, P02*P23 + P03*P22, P12*P23 + P13*P22, -P22*P23};
    VECTORIZE_LOOP
    for(int i = 0; i < num_pairs; i++)
    {
        const Scalar x0 = X0[i], y0 = Y0[i], x1 = X1[i], y1 = Y1[i];
        const Scalar term[8] = {x0*x0, y0*y0, x1*x1, y1*y1, x0, y0, x1, y1};
        const Scalar A00 = term[2] * CA00[0] + CA00[1] + term[6] * CA00[2] + term[7] * CA00[3] + term[3] * CA00[4];
        const Scalar A01 = term[2] * CA01[0] + CA01[1] + term[6] * CA01[2] + term[7] * CA01[3] + term[3] * CA01[4];
        const Scalar A02 = CA02[0] + term[6] * CA02[1] + term[2] * CA02[2] + term[7] * CA02[3] + term[3] * CA02[4] + term[4] * CA02[5];
        const Scalar A10 = term[2] * CA10[0] + CA10[1] + term[6] * CA10[2] + term[7] * CA10[3] + term[3] * CA10[4];
        const Scalar A11 = term[2] * CA11[0] + CA11[1] + term[6] * CA11[2] + term[7] * CA11[3] + term[3] * CA11[4];
        const Scalar A12 = CA12[0] + term[6] * CA12[1] + term[5] * CA12[2] + term[2] * CA12[3] + term[7] * CA12[4] + term[3] * CA12[5];
        const Scalar A20 = CA20[0] + term[6] * CA20[1] + term[2] * CA20[2] + term[7] * CA20[3] + term[3] * CA20[4] + term[4] * CA20[5];
        const Scalar A21 = CA21[0] + term[6] * CA21[1] + term[5] * CA21[2] + term[2] * CA21[3] + term[7] * CA21[4] + term[3] * CA21[5];
        const Scalar A22 = CA22[0] + term[6] * CA22[1] + term[1] * CA22[2] + term[2] * CA22[3] + term[0] * CA22[4] + term[7] * CA22[5] + term[4] * CA22[6] + term[3] * CA22[7] + term[5] * CA22[8];
        const Scalar b0 = term[2] * Cb0[0] + Cb0[1] + term[6] * Cb0[2] + term[7] * Cb0[3] + term[3] * Cb0[4];
        const Scalar b1 = term[2] * Cb1[0] + Cb1[1] + term[6] * Cb1[2] + term[7] * Cb1[3] + term[3] * Cb1[4];
        const Scalar b2 = term[2] * Cb2[0] + Cb2[1] + term[6] * Cb2[2] + term[7] * Cb2[3] + term[3] * Cb2[4];

        const Scalar A11A22_A12A21 = A11*A22 - A12*A21;
        const Scalar A10A22 = A10*A22, A12A20 = A12*A20, A10A21 = A10*A21, A11A20 = A11*A20;
        const Scalar x = b0*(A11A22_A12A21) + b1*(-A01*A22 + A02*A21) + b2*(A01*A12 - A02*A11);
        const Scalar y = -b0*(A10A22 - A12A20) - b1*(-A00*A22 + A02*A20) - b2*(A00*A12 - A02*A10);
        const Scalar z = b0*(A10A21 - A11A20) + b1*(-A00*A21 + A01*A20) + b2*(A00*A11 - A01*A10);
        const Scalar scale = 1.0 / (A00*(A11A22_A12A21) - A01*(A10A22 - A12A20) + A02*(A10A21 - A11A20));
        vec_x[i] = x * scale; vec_y[i] = y * scale; vec_z[i] = z * scale;
    }
    Eigen::Matrix<Scalar, rowsAtCompileTime, 3> result(num_pairs, 3);
    result << vec_x, vec_y, vec_z;

    const Eigen::Matrix<bool, rowsAtCompileTime, 1> sanity = result.array().isFinite().rowwise().all();
    const Scalar fx0_inv = 1.f / fx0;
    const Scalar fy0_inv = 1.f / fy0;
    for(int i = 0; i < num_pairs; i++){
        if(std::abs(result(i, 2)) < 1E-10f)
            result(i, 2) = 1E-3f;
        if(!sanity[i])
        {
            Vector2<Scalar> pt2d;
            pt2d << pairs[0][i], pairs[1][i];
            Vector3<Scalar> pt3d;
            pt3d << (pt2d[0] - cx0) * fx0_inv, (pt2d[1] - cy0) * fy0_inv, 1.f;
            result.row(i) = Scalar(1000.f) * pt3d.transpose();
        }
    }

    return result;
};
#else
Eigen::MatrixX3f triangulate(
        const Camera &camera0, const Camera &camera1, const Eigen::Isometry3f &RT,
        const std::array<Eigen::ArrayXf, 4> &pairs);
#endif

inline Eigen::Array<bool, Eigen::Dynamic, 1> checkSim3(const Eigen::Matrix<float, 3, 4>& trans, const std::array<Eigen::ArrayXf, 3>& src,
    const std::array<Eigen::ArrayXf, 3>& dst, const float threshold)
{
    return (
        (trans(0, 0) * src[0] + trans(0, 1) * src[1] + trans(0, 2) * src[2]  + trans(0, 3) - dst[0]).square() +
        (trans(1, 0) * src[0] + trans(1, 1) * src[1] + trans(1, 2) * src[2]  + trans(1, 3) - dst[1]).square() +
        (trans(2, 0) * src[0] + trans(2, 1) * src[1] + trans(2, 2) * src[2]  + trans(2, 3) - dst[2]).square() < threshold * threshold).eval();
}

template <bool isNormalzied>
Eigen::Quaternionf findMinRotation(const Eigen::Vector3f& src, const Eigen::Vector3f& dst)
{
    assert(!isNormalzied || (std::abs(src.norm() - 1) < 1E-4f && std::abs(dst.norm() - 1) < 1E-4f));
    const Eigen::Vector3f s = isNormalzied ? src : src.normalized();
    const Eigen::Vector3f d = isNormalzied ? dst : dst.normalized();
    const Eigen::Vector3f crossProd = s.cross(d);
    const float sinTheta = crossProd.norm();
    const float dotProd = s.dot(d);
    const Eigen::Vector3f axis = crossProd * (1.f / sinTheta);
    const float angle = std::atan2(sinTheta, dotProd);
    const Eigen::Quaternionf q{Eigen::AngleAxisf{angle, axis}};
    assert(!((q*s - d).norm() > 1E-3f));
    return q;
}

//@fixme: need tests
//@fixme: this method may not be very accurate
// each row of src/dst is a sample
inline Sim3Transform findSim3(const Eigen::Matrix3f& srcSamples, const Eigen::Matrix3f&  dstSamples)
{
    Eigen::Matrix3f src = srcSamples;
    Eigen::Matrix3f dst = dstSamples;
    const Eigen::RowVector3f srcCenter = src.colwise().mean();
    const Eigen::RowVector3f dstCenter = dst.colwise().mean();
    src = src.rowwise() - srcCenter;
    dst = dst.rowwise() - dstCenter;
    const float srcLen = std::sqrt(src.rowwise().squaredNorm().sum());
    const float dstLen = std::sqrt(dst.rowwise().squaredNorm().sum());
    const float scale = dstLen / srcLen;
    src *= scale;

    // rotate to match the first point
    const Eigen::Quaternionf rotation1 = findMinRotation<false>(src.row(0), dst.row(0));
    src = (rotation1 * src.transpose()).transpose();
    // rotate to match the second point with axis being the first point.
    const Eigen::Vector3f axis2 = dst.row(0).transpose().normalized();
    auto makePerpVec = [&](const Eigen::Vector3f& v) -> Eigen::Vector3f {
        return axis2.cross(v).cross(axis2).eval();
    };
    const Eigen::Quaternionf rotation2 = findMinRotation<false>(makePerpVec(src.row(1).transpose()), makePerpVec(dst.row(1).transpose()));

    const Eigen::Quaternionf rotation = rotation2 * rotation1;
    const Eigen::Matrix3f linear = scale * rotation.toRotationMatrix();
    const Eigen::Vector3f translation = dstCenter.transpose() - linear * srcCenter.transpose();

	return Sim3Transform{.R = fromEigen(rotation), .scale = scale, .t = fromEigen(translation)};
}

inline int64_t countInliers(const std::vector<bool>& mask) {
    return std::count(mask.begin(), mask.end(), true);
}

Eigen::Matrix<float, Eigen::Dynamic, 4> mask2pts(const std::array<Eigen::ArrayXf, 4>& ptPairs, const bool* mask);

} // namespace rsfm
