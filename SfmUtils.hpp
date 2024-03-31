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
#include "SfmUtils.h"
#include <eigen3/Eigen/Dense>
#include "Types.hpp"

namespace rsfm
{
enum class LeastSquareSolverType{
    kSVD, kQR, kNormalEquation
};

template<typename Derived1, typename Derived2, LeastSquareSolverType solver_type = LeastSquareSolverType::kNormalEquation>
Eigen::Matrix<typename Derived1::Scalar, Derived1::ColsAtCompileTime, Derived2::ColsAtCompileTime>
    leastSquareSolve(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& b)
{
    Eigen::Matrix<typename Derived1::Scalar, Derived1::ColsAtCompileTime, Derived2::ColsAtCompileTime> X(A.cols(), b.cols());

    if(A.rows() == A.cols() && A.rows() < 4)
    {
        X = A.inverse() * b;
    }
    else{
        switch(solver_type)
        {
        case LeastSquareSolverType::kSVD:
            X = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b).eval();
            break;
        case LeastSquareSolverType::kQR:
            X = A.householderQr().solve(b).eval();
            break;
        case LeastSquareSolverType::kNormalEquation:
            X = (A.transpose() * A).eval().llt().solve((A.transpose() * b).eval()).eval();
            break;
        default:
            throw std::runtime_error("fatal error");
        }
    }
    return X;
}


template<typename T>
Matrix3<T> gvec2mat(const T gvec[3])
{
    const T& g0 = gvec[0];
    const T& g1 = gvec[1];
    const T& g2 = gvec[2];
    const T factor = 2 / (g0*g0 + g1*g1 + g2*g2 + 1);
    Eigen::Matrix<T, 3, 3> R;
    R <<    g0*g0 + 1,   g0*g1 - g2, g0*g2 + g1,
            g0*g1 + g2,  g1*g1 + 1,  g1*g2 - g0,
            g0*g2 - g1,  g1*g2 + g0, g2*g2 + 1;
    R = R * factor;
    R.diagonal().array() -= 1.f;

    return R;
}

template<typename T>
Matrix3<T> gvec2mat(const Vector3<T>& gvec){
    return gvec2mat(&gvec[0]);
};

template<typename Derived>
Vector3<typename Derived::Scalar> mat2gvec(const Eigen::MatrixBase<Derived>& R){
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 3, "input must be 3x3");
    Vector3<typename Derived::Scalar> g;
    g << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    g *= 1 / (R(0, 0) + R(1, 1) + R(2, 2) + 1);
    assert((gvec2mat(g) - R).array().abs().eval().maxCoeff() < 1E-4f);
    return g;
}

// b * a, i.e. apply a first, then apply b rotation, equivalent to gvec2mat(b) * gvec2mat(a)
template<typename Derived>
Vector3<typename Derived::Scalar> gvec_mul(const Eigen::MatrixBase<Derived>& b, const Eigen::MatrixBase<Derived>& a) {
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "input must be 3x1");
    return (a + b + b.cross(a)) / (1 - a.dot(b));
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        orthogonalised(const Eigen::MatrixBase<Derived>& mat) {
#if 0 // via gvec, only works for 3x3
    return gvec2mat(mat2gvec(mat));
#else // by SVD
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
#endif
};

template <typename Derived1, typename Derived2>
bool approx(const Eigen::MatrixBase<Derived1>& a, const Eigen::MatrixBase<Derived2>& b, const typename Derived1::Scalar threshold = 1E-4f){
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    return ((a - b).array().abs() < threshold * b.array().abs().cwiseMax(1)).all();
}

template <typename Derived1, typename Derived2>
bool approx(const Eigen::ArrayBase<Derived1>& a, const Eigen::ArrayBase<Derived2>& b, const typename Derived1::Scalar threshold = 1E-4f){
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    return ((a - b).array().abs() < threshold * b.array().abs().cwiseMax(1)).all();
}

//find the minimum rotation from [0,0,1] to T.
//@todo: add unit test
//Tgvec[2] == 0.f
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> T2Tgvec(const Eigen::MatrixBase<Derived>& T){
    typedef typename Derived::Scalar scalar_type;
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "invalid T vector");
//    scalar_type xy_sqr_norm = T.template topRows<2>().squaredNorm();
//    scalar_type tan_half_theta_div_xy_norm = (T.norm() - T[2]) / xy_sqr_norm;
//T(2) == -T.norm() correspond to 180degree rotation
    const scalar_type tan_half_theta_div_xy_norm = 1.f/(T.norm() + T[2]);
    Eigen::Matrix<scalar_type, 3, 1> Tgvec;
    Tgvec << -T[1] * tan_half_theta_div_xy_norm, T[0] * tan_half_theta_div_xy_norm, 0.f;
    assert((gvec2mat(Tgvec) * (Eigen::Matrix<scalar_type, 3, 1>() << 0, 0, 1).finished() - T).array().abs().max() < 1E-3f);
    return Tgvec;
}
//optimised version for T.norm() == 1
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> UnitT2Tgvec(const Eigen::MatrixBase<Derived>& T){
    assert(std::abs(T.norm() - 1) < 1E-4f);
    typedef typename Derived::Scalar scalar_type;
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "invalid T vector");
    const scalar_type tan_half_theta_div_xy_norm = 1.f/(1 + T[2]);
    Eigen::Matrix<scalar_type, 3, 1> Tgvec;
    Tgvec << -T[1] * tan_half_theta_div_xy_norm, T[0] * tan_half_theta_div_xy_norm, 0.f;
    assert(((gvec2mat(Tgvec) * (Eigen::Matrix<scalar_type, 3, 1>() << 0, 0, 1).finished() - T).array().abs() < 1E-3f).all());
    return Tgvec;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> T2TR(const Eigen::MatrixBase<Derived>& T){
    return gvec2mat(T2Tgvec(T));
}


template<typename Derived1, typename Derived2>
inline auto solveGaussElim(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& b){
    typedef typename Derived1::Scalar ElemType;
    static_assert(std::is_same<typename Derived1::Scalar, typename Derived2::Scalar>::value, "fatal error");
    assert(A.rows() == b.rows() && A.rows() == A.cols());
    Eigen::Matrix<ElemType, std::decay_t<decltype(A)>::RowsAtCompileTime, (std::decay_t<decltype(A)>::ColsAtCompileTime < 0 || std::decay_t<decltype(b)>::ColsAtCompileTime < 0) ? -1 : int(std::decay_t<decltype(A)>::ColsAtCompileTime) + int(std::decay_t<decltype(b)>::ColsAtCompileTime)> Ab(A.rows(), A.cols() + b.cols());
    for(int r = 0; r < A.rows(); r++){
        Ab.row(r) << A.row(r), b.row(r);
    }
    for (int i = 0; i < A.rows(); i++) {
        const ElemType inv = 1.f / Ab(i, i);
        for (int j = i + 1; j < A.rows(); j++) {
            const ElemType factor = Ab(j, i) * inv;
            for (int k = i + 1; k < A.cols() + b.cols(); k++) {
                Ab(j, k) -= factor * Ab(i, k);
            }
        }
    }
    Eigen::Matrix<ElemType, std::decay_t<decltype(b)>::RowsAtCompileTime, std::decay_t<decltype(b)>::ColsAtCompileTime> x = Ab.template rightCols<std::decay_t<decltype(b)>::ColsAtCompileTime>();
    for (int i = A.rows() - 1; i >= 0; i--) {
        const ElemType inv = 1.f / Ab(i, i);
        x.row(i) *= inv;
        for (int j = i - 1; j >= 0; j--){
            const ElemType factor = Ab(j, i);
            x.row(j) -= factor * x.row(i);
        }
    }
    return x;
}

template <size_t nbSamples, size_t nbDims>
Eigen::Matrix<float, nbSamples, nbDims> sampleWithIndices(const std::array<Eigen::ArrayXf, nbDims>& allData, const std::array<uint32_t, nbSamples>& indices) {
    Eigen::Matrix<float, nbSamples, nbDims> samples;
    for(unsigned i = 0; i < nbSamples; i++)
    {
        const uint32_t idx = indices[i];
        for (unsigned j = 0; j < nbDims; j++) {
            samples(i, j) = allData[j][idx];
        }
    }
    return samples;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <size_t nbDims>
Eigen::Matrix<float, Eigen::Dynamic, nbDims> sampleWithIndices(const std::array<Eigen::ArrayXf, nbDims>& allData, const uint32_t* indices, size_t nbSamples) {
    Eigen::Matrix<float, Eigen::Dynamic, nbDims> samples(static_cast<int>(nbSamples), nbDims);
    for(unsigned i = 0; i < nbSamples; i++)
    {
        const uint32_t idx = indices[i];
        for (unsigned j = 0; j < nbDims; j++) {
            samples(i, j) = allData[j][idx];
        }
    }
    return samples;
}
#pragma GCC diagnostic pop

template <int32_t nbSamples, int32_t nbRandSamples, int32_t nbDims>
Eigen::Matrix<float, nbSamples, nbDims> concatSamples(const Eigen::Matrix<float, nbRandSamples, nbDims>& randSamples, const std::array<Eigen::ArrayXf, nbDims>& tiePtObPairs) {
	const auto nbTiePts = tiePtObPairs.at(0).size();
	ASSERT(nbSamples == Eigen::Dynamic || nbSamples == randSamples.rows() + nbTiePts);
	if constexpr (nbSamples == nbRandSamples && nbSamples != Eigen::Dynamic) {
		return randSamples;
	}
	else {
		Eigen::Matrix<float, nbSamples, nbDims> result(randSamples.rows() + nbTiePts, nbDims);
		if (nbTiePts == 0) {
			result << randSamples;
		}
		else {
			result << randSamples, tiePtObPairs[0], tiePtObPairs[1], tiePtObPairs[2], tiePtObPairs[3];
		}
		return result;
	}
}

template <int nbSamples, int nbDims>
Eigen::Matrix<float, nbDims, nbDims> computeCovariance (const Eigen::Matrix<float, nbSamples, nbDims>& samples){
        assert(samples.rows() >= nbDims && nbDims > 1);
        std::array<Eigen::VectorXf, nbDims> cols;
        for (int i = 0; i < nbDims; i++) {
            cols[i] = samples.col(i).array() - samples.col(i).mean();
        }
        Eigen::Matrix<float, nbDims, nbDims> covariance;
        for (int i = 0; i < nbDims; i++) {
            for (int j = 0; j < i; j++) {
                covariance(i, j) = covariance(j, i) = cols[i].dot(cols[j]);
            }
            covariance(i, i) = cols[i].squaredNorm();
        }
        const float scale = 1.f / (samples.rows() - 1);
        covariance *= scale;
        return covariance;
    };
}
