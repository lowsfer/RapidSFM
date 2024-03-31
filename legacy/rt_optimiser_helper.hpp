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
// Created by yao on 16/11/17.
//

#pragma once
#include "rt_optimiser.hpp"
#include "../Types.hpp"
#include <cpp_utils.h>

#pragma GCC diagnostic push
#ifdef NDEBUG
#pragma GCC optimize("-ffast-math", "-fno-finite-math-only")
#endif
namespace rsfm::legacy
{

template<int RowsAtCompileTime, int ColsAtCompileTime>
void set_padding(Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, const unsigned count, float value)
{
    assert(count < rt_optimiser::array_type::size);
    for(int i = 0; i < array_mat.rows(); i++)
        for(int j = 0; j < array_mat.cols(); j++)
            array_mat(i, j).data.bottomRows(rt_optimiser::array_type::size - count).setConstant(value);
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
void set_nan_padding(Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, const unsigned count)
{
    set_padding(array_mat, count, NAN);
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
void set_zero_padding(Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, const unsigned count)
{
    set_padding(array_mat, count, 0);
}

// This check cannot be done when we use finite math (included in -ffast-math)
template<int RowsAtCompileTime, int ColsAtCompileTime>
bool check_nan_padding(const Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, const unsigned count)
{
#ifndef __FAST_MATH__
    assert(count <= rt_optimiser::array_type::size);
    if(count == rt_optimiser::array_type::size)
        return true;
    for(int i = 0; i < array_mat.rows(); i++){
        for(int j = 0; j < array_mat.cols(); j++){
            if(!(array_mat(i, j).data.isNaN().bottomRows(rt_optimiser::array_type::size - count)).all()){
                return false;
            }
        }
    }
#endif
    return true;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
bool check_zero_padding(const Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, const unsigned count)
{
    assert(count <= rt_optimiser::array_type::size);
    if(count == rt_optimiser::array_type::size)
        return true;
    for(int i = 0; i < array_mat.rows(); i++)
        for(int j = 0; j < array_mat.cols(); j++)
            if(!(array_mat(i, j).data.bottomRows(rt_optimiser::array_type::size - count) == 0).all())
                return false;
    return true;
}

template<int RowsAtCompileTime, int ColsAtCompileTime, typename PadChecker>
bool check_array_mat_padding(const EigenAlignedVector<Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>>& array_mat, const unsigned count, const PadChecker& checker)
{
    assert(rt_optimiser::array_type::size * array_mat.size() >= count);
    const unsigned residue = unsigned(count % rt_optimiser::array_type::size);
    if(residue != 0){
        const auto& last_block = array_mat.at(count / rt_optimiser::array_type::size);
        checker(last_block, residue);
    }
    return true;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
bool check_array_mat_zero_padding(const EigenAlignedVector<Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>>& array_mat, const unsigned count)
{
    return check_array_mat_padding(array_mat, count, &check_zero_padding<RowsAtCompileTime, ColsAtCompileTime>);
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
bool check_array_mat_nan_padding(const EigenAlignedVector<Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>>& array_mat, const unsigned count)
{
    return check_array_mat_padding(array_mat, count, &check_nan_padding<RowsAtCompileTime, ColsAtCompileTime>);
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
bool check_all_finite(const Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat,
                      const unsigned count = rt_optimiser::array_type::size)
{
    assert(count <= rt_optimiser::array_type::size);
    for(int i = 0; i < array_mat.rows(); i++)
        for(int j = 0; j < array_mat.cols(); j++)
            if(!array_mat(i, j).data.topRows(count).allFinite())
                return false;
    return true;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime>
array_mat_sum(const Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat)
{
    Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime> result(array_mat.rows(), array_mat.cols());
    for(int i = 0; i < array_mat.rows(); i++)
        for(int j = 0; j < array_mat.cols(); j++)
            result(i, j) = array_mat(i, j).sum();
    return result;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime>
array_mat_sum(const Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>& array_mat, unsigned count)
{
    assert(count <= rt_optimiser::array_type::size);
    Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime> result(array_mat.rows(), array_mat.cols());
    for(int i = 0; i < array_mat.rows(); i++)
        for(int j = 0; j < array_mat.cols(); j++)
            result(i, j) = array_mat(i, j).topRows(count).sum();
    return result;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime>
array_mat_sum(const EigenAlignedVector<Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>>& array_mat, const size_t count)
{
    assert(rt_optimiser::array_type::size * array_mat.size() >= count);
    Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime> partial_sum;
    partial_sum.setConstant(rt_optimiser::array_type(0));
    const unsigned num_whole_blocks = unsigned(count / rt_optimiser::array_type::size);
    for(unsigned n = 0; n < num_whole_blocks; n++){
        partial_sum += array_mat[n];
    }
    const unsigned residue = unsigned(count % rt_optimiser::array_type::size);
    if(residue != 0){
        const auto& last_block = array_mat[num_whole_blocks];
        for(int i = 0; i < partial_sum.rows(); i++)
            for(int j = 0; j < partial_sum.cols(); j++)
                partial_sum(i, j).data.topRows(residue) += last_block(i, j).data.topRows(residue);
    }

    Eigen::Matrix<rt_optimiser::fptype, RowsAtCompileTime, ColsAtCompileTime> result(partial_sum.rows(), partial_sum.cols());
    for(int i = 0; i < partial_sum.rows(); i++)
        for(int j = 0; j < partial_sum.cols(); j++)
            result(i, j) = partial_sum(i, j).sum();
    return result;
}

template<int RowsAtCompileTime, int ColsAtCompileTime>
rt_optimiser::fptype array_mat_max(const EigenAlignedVector<Eigen::Matrix<rt_optimiser::array_type, RowsAtCompileTime, ColsAtCompileTime>>& array_mat, const unsigned i, const unsigned j, const size_t count)
{
    assert(rt_optimiser::array_type::size * array_mat.size() >= count);
    assert(i < array_mat[0].rows() && j < array_mat[0].cols());
    rt_optimiser::array_type partial_max(std::numeric_limits<rt_optimiser::fptype>::lowest());
    const unsigned num_whole_blocks = unsigned(count / rt_optimiser::array_type::size);
    for(unsigned n = 0; n < num_whole_blocks; n++){
        partial_max.data.cwiseMax(array_mat[n](i,j).data);
    }
    const unsigned residue = unsigned(count % rt_optimiser::array_type::size);
    if(residue != 0){
        const auto& last_block = array_mat[num_whole_blocks](i, j);
        partial_max.data.topRows(residue).cwiseMax(last_block.data.topRows(residue));
    }
    return partial_max.max();
}

class error_jacobian_calculator{
public:
    using fptype = rt_optimiser::fptype;
    using array_type = rt_optimiser::array_type;
public:
    error_jacobian_calculator(
            const PinHoleCamera& cam_left, const PinHoleCamera& cam_right,
            const Isometry3<fptype>& RT)
            :cam_left(cam_left), cam_right(cam_right), RT(RT){
    }

private:
    const PinHoleCamera cam_left;
    const PinHoleCamera cam_right;
    const Isometry3<fptype> RT;

    const Eigen::Matrix<fptype, 3, 4> P = toKMat<fptype>(cam_right) * RT.matrix().template topRows<3>();
    const Eigen::Array<fptype, 3, 3> Jxyz_pt = P.template leftCols<3>();
    const Matrix3<fptype> TR2 = 2 * gvec2mat(UnitT2Tgvec(RT.translation()));
    const Matrix3<fptype> R2 = 2 * RT.linear();
    const Eigen::Matrix<fptype, 3, 4> P2 = 2 * P;
    //homogeneous coordinate x,y,z derivative of tg0, tg1
    const Eigen::Array<fptype, 3, 2> Jxyz_camT = [](const auto& cam_right, const auto& TR2){
        const fptype fx = cam_right.f.x;
        const fptype fy = cam_right.f.y;
        const fptype cx = cam_right.c.x;
        const fptype cy = cam_right.c.y;
        Eigen::Array<fptype, 3, 2> result;
        result <<
                -TR2(0,1)*fx - TR2(2,1)*cx,  TR2(0,0)*fx + TR2(2,0)*cx,
                -TR2(1,1)*fy - TR2(2,1)*cy,  TR2(1,0)*fy + TR2(2,0)*cy,
                -TR2(2,1),                   TR2(2,0);
        return result;
    }(cam_right, TR2);

    Eigen::Matrix<array_type, 5, 5> U_acc = Eigen::Matrix<array_type, 5, 5>::Constant(array_type(0));
    Eigen::Matrix<array_type, 5, 1> g_a_acc = Eigen::Matrix<array_type, 5, 1>::Constant(array_type(0));
    array_type robust_error_sqrNorm_acc = array_type(0);
    array_type g_b_absmax = array_type(0);

public:
    template<bool use_huber, bool useWeights, bool is_full>
    void calculate_error_jacobian(
            const Eigen::Matrix<array_type, 3, 1>& pts3d,
            const Eigen::Matrix<array_type, 2, 1>& left,
            const Eigen::Matrix<array_type, 2, 1>& right,
			const array_type* pHuberL,
			const array_type* pHuberR,
			const array_type* weightLeft,
			const array_type* weightRight,
            Eigen::Matrix<array_type, 3, 3>& V,
            Eigen::Matrix<array_type, 5, 3>& W,
            Eigen::Matrix<array_type, 3, 1>& g_b,
            unsigned count = array_type::size
    ){
        Eigen::Matrix<array_type, 2, 1> error_left;
        Eigen::Matrix<array_type, 2, 1> error_right;
        Eigen::Matrix<array_type, 2, 3> jacobian_left_pt;
        Eigen::Matrix<array_type, 2, 5> jacobian_right_cam;
        Eigen::Matrix<array_type, 2, 3> jacobian_right_pt;
        Eigen::Matrix<array_type, 2, 1> rho_left;
        Eigen::Matrix<array_type, 2, 1> rho_right;
        calculate_error_jacobian<use_huber, useWeights, is_full>(
                pts3d, left, right, pHuberL, pHuberR, weightLeft, weightRight,
				error_left, error_right,
                jacobian_left_pt, jacobian_right_cam, jacobian_right_pt,
                V, W, g_b,
                use_huber ? &rho_left : nullptr,
                use_huber ? &rho_right : nullptr,
                count);
		assert(check_all_finite(error_left, count));
		assert(check_all_finite(error_right, count));
        if(!is_full && count != array_type::size){
            assert(check_zero_padding(error_left, count));
            assert(check_zero_padding(error_right, count));
            assert(check_zero_padding(jacobian_left_pt, count));
            assert(check_zero_padding(jacobian_right_cam, count));
            assert(check_zero_padding(jacobian_right_pt, count));
            if(use_huber) {
                assert(check_zero_padding(rho_left, count));
                assert(check_zero_padding(rho_right, count));
            }
            assert(check_zero_padding(V, count));
            assert(check_zero_padding(W, count));
            assert(check_zero_padding(g_b, count));
        }
    }

    template<bool use_huber, bool useWeights, bool is_full>
    void calculate_error_jacobian(
            const Eigen::Matrix<array_type, 3, 1>& pts3d,
            const Eigen::Matrix<array_type, 2, 1>& left,
            const Eigen::Matrix<array_type, 2, 1>& right,
			const array_type* pHuberL,
			const array_type* pHuberR,
			const array_type* weightLeft,
			const array_type* weightRight,
            Eigen::Matrix<array_type, 2, 1>& error_left,
            Eigen::Matrix<array_type, 2, 1>& error_right,
            Eigen::Matrix<array_type, 2, 3>& jacobian_left_pt,
            Eigen::Matrix<array_type, 2, 5>& jacobian_right_cam,
            Eigen::Matrix<array_type, 2, 3>& jacobian_right_pt,
            Eigen::Matrix<array_type, 3, 3>& V,
            Eigen::Matrix<array_type, 5, 3>& W,
            Eigen::Matrix<array_type, 3, 1>& g_b,
            Eigen::Matrix<array_type, 2, 1>* rho_left/* = nullptr*/,
            Eigen::Matrix<array_type, 2, 1>* rho_right/* = nullptr*/,
            unsigned count = array_type::size
    ){
        assert((!is_full && count <= array_type::size)
               || (is_full && count == array_type::size));
        assert(std::abs(RT.translation().norm() - 1) < 1E-5f);
        const bool need_padding = (!is_full && count != array_type::size);

        {//left
            const array_type z_inv(1 / pts3d[2].data);
            error_left << pts3d[0] * cam_left.f.x * z_inv + cam_left.c.x - left[0],
                    pts3d[1] * cam_left.f.y * z_inv + cam_left.c.y - left[1];
            jacobian_left_pt << cam_left.f.x * z_inv, array_type(0), -cam_left.f.x * pts3d[0] * square(z_inv),
                    array_type(0), cam_left.f.y * z_inv, -cam_left.f.y * pts3d[1] * square(z_inv);
            assert(check_all_finite(error_left, count));
            assert(check_all_finite(jacobian_left_pt, count));
            if(use_huber){
				const array_type::data_type& huber_delta = pHuberL->data;
				const array_type::data_type huber_delta_x2 = huber_delta * 2;
				const array_type::data_type sqr_huber_delta = square(huber_delta);
                const array_type::data_type e = useWeights ? error_left.squaredNorm().data * weightLeft->data : error_left.squaredNorm().data;
                const Eigen::Array<bool, array_type::size, 1> mask_left = (e < sqr_huber_delta).eval();
                if(!mask_left.all()) {
                    const array_type::data_type sqrte_left = e.cwiseSqrt();
                    (*rho_left)[0].data = mask_left.select(e, huber_delta_x2 * sqrte_left - sqr_huber_delta);
                    (*rho_left)[1].data = mask_left.select(array_type::data_type::Ones(), huber_delta / sqrte_left);
                    assert(check_all_finite((*rho_left), count));
                }
				else{
                    (*rho_left)[0].data = e;
                    (*rho_left)[1].data.setConstant(1);
                }
            }
            if(need_padding){
                set_zero_padding(error_left, count);
                set_zero_padding(jacobian_left_pt, count);
                if(use_huber)
                    set_zero_padding(*rho_left, count);
            }
        }
        {//right
            const array_type x = pts3d[0] * P(0,0) + pts3d[1] * P(0,1) + pts3d[2] * P(0,2) + P(0, 3);
            const array_type y = pts3d[0] * P(1,0) + pts3d[1] * P(1,1) + pts3d[2] * P(1,2) + P(1, 3);
            const array_type z = pts3d[0] * P(2,0) + pts3d[1] * P(2,1) + pts3d[2] * P(2,2) + P(2, 3);
            const array_type z_inv = fptype(1) / z;
            error_right << x * z_inv - right[0], y * z_inv - right[1];

            {
                const array_type& px = pts3d[0];
                const array_type& py = pts3d[1];
                const array_type& pz = pts3d[2];

                Eigen::Array<array_type, 3, 3> Jxyz_camG;//homogeneous coordinate x,y,z derivative of g0, g1, g2
                Jxyz_camG <<
                        py*P2(0,2) - pz*P2(0,1), pz*P2(0,0) - px*P2(0,2), px*P2(0,1) - py*P2(0,0),
                        py*P2(1,2) - pz*P2(1,1), pz*P2(1,0) - px*P2(1,2), px*P2(1,1) - py*P2(1,0),
                        py*R2(2,2) - pz*R2(2,1), pz*R2(2,0) - px*R2(2,2), px*R2(2,1) - py*R2(2,0);
                const array_type z_inv_sqr_neg = -square(z_inv);
                const array_type tmp[2] = {x * z_inv_sqr_neg, y * z_inv_sqr_neg};
                jacobian_right_cam
                        << z_inv * Jxyz_camG.row(0) + tmp[0] * Jxyz_camG.row(2), z_inv * Jxyz_camT(0, 0) + tmp[0] * Jxyz_camT(2, 0), z_inv * Jxyz_camT(0, 1) + tmp[0] * Jxyz_camT(2, 1),
                        z_inv * Jxyz_camG.row(1) + tmp[1] * Jxyz_camG.row(2), z_inv * Jxyz_camT(1, 0) + tmp[1] * Jxyz_camT(2, 0), z_inv * Jxyz_camT(1, 1) + tmp[1] * Jxyz_camT(2, 1);
                for(int i = 0; i < 2; i++)
                    for(int j = 0; j < 3; j++)
                        jacobian_right_pt(i,j) = z_inv * Jxyz_pt(i,j) + tmp[i] * Jxyz_pt(2, j);
            }
            if(use_huber){
				const array_type::data_type& huber_delta = pHuberR->data;
				const array_type::data_type huber_delta_x2 = huber_delta * 2;
				const array_type::data_type sqr_huber_delta = square(huber_delta);
                const array_type::data_type e = useWeights ? error_right.squaredNorm().data * weightRight->data : error_right.squaredNorm().data;
                const Eigen::Array<bool, array_type::size, 1> mask_right = (e < sqr_huber_delta).eval();
                if(!mask_right.all()) {
                    const array_type::data_type sqrte_right = e.cwiseSqrt();
                    (*rho_right)[0].data = mask_right.select(e, huber_delta_x2 * sqrte_right - sqr_huber_delta);
                    (*rho_right)[1].data = mask_right.select(array_type::data_type::Ones(), huber_delta / sqrte_right);
                    assert(check_all_finite((*rho_right), count));
                }
				else{
                    (*rho_right)[0].data = e;
                    (*rho_right)[1].data.setConstant(1);
                }
            }
            if(need_padding){
                set_zero_padding(error_right, count);
                set_zero_padding(jacobian_right_cam, count);
                set_zero_padding(jacobian_right_pt, count);
                if(use_huber)
                    set_zero_padding(*rho_right, count);
            }
        }
        {//UVWg
            const auto &A = jacobian_right_cam;
            const auto &B_left = jacobian_left_pt;
            const auto &B_right = jacobian_right_pt;
            if(use_huber){
				const array_type robustWeightLeft = useWeights ? *weightLeft * (*rho_left)[1] : (*rho_left)[1];
				const array_type robustWeightRight = useWeights ? *weightRight * (*rho_right)[1] : (*rho_right)[1];
                U_acc += robustWeightRight * A.transpose() * A;
                V = robustWeightLeft * B_left.transpose() * B_left + robustWeightRight * B_right.transpose() * B_right;
                W = A.transpose() * (robustWeightRight * B_right);
                robust_error_sqrNorm_acc += (*rho_left)[0] * robustWeightLeft;
                robust_error_sqrNorm_acc += (*rho_right)[0] * robustWeightRight;
                g_a_acc += A.transpose() * (robustWeightRight * error_right);
                g_b = B_left.transpose() * (robustWeightLeft * error_left) +
                        B_right.transpose() * (robustWeightRight * error_right);
            }else {
				if (useWeights) {
					U_acc += *weightRight * (A.transpose() * A);
					V = *weightLeft * B_left.transpose() * B_left + *weightRight * B_right.transpose() * B_right;
					W = A.transpose() * (*weightRight * B_right);
					robust_error_sqrNorm_acc += error_left.squaredNorm() * *weightLeft;
					robust_error_sqrNorm_acc += error_right.squaredNorm() * *weightRight;
					g_a_acc += A.transpose() * (*weightRight * error_right);
					g_b = B_left.transpose() * (*weightLeft * error_left) + B_right.transpose() * (*weightRight * error_right);
				}
				else {
					U_acc += (A.transpose() * A);
					V = B_left.transpose() * B_left + B_right.transpose() * B_right;
					W = A.transpose() * B_right;
					robust_error_sqrNorm_acc += error_left.squaredNorm();
					robust_error_sqrNorm_acc += error_right.squaredNorm();
					g_a_acc += A.transpose() * error_right;
					g_b = B_left.transpose() * error_left + B_right.transpose() * error_right;
				}
            }
            for (int i = 0; i < g_b.size(); i++)
                g_b_absmax.data = g_b_absmax.data.cwiseMax(g_b[i].data.abs());
            if(need_padding){
                assert(check_zero_padding(A, count));
                assert(check_zero_padding(B_left, count));
                assert(check_zero_padding(B_right, count));
                assert(check_zero_padding(V, count));
                assert(check_zero_padding(W, count));
                assert(check_zero_padding(g_b, count));
            }
        }
    }
    //get accumulated U, g_a and robust error norm
    void get_UgE(Eigen::Matrix<fptype, 5, 1> &g_a, Eigen::Matrix<fptype, 5, 5> &U, fptype &error_norm_robust, fptype &g_absmax) const{
        assert(check_all_finite(U_acc));
        assert(check_all_finite(g_a_acc));
        assert(robust_error_sqrNorm_acc.data.allFinite());
        assert(g_b_absmax.data.allFinite());

        U = array_mat_sum(U_acc);
        g_a = array_mat_sum(g_a_acc);
        error_norm_robust = std::sqrt(robust_error_sqrNorm_acc.sum());
        g_absmax = std::max(g_a.cwiseAbs().maxCoeff(), g_b_absmax.max());
    }
};

class error_calculator{
public:
    using fptype = rt_optimiser::fptype;
    using array_type = rt_optimiser::array_type;
public:
    error_calculator(
            const PinHoleCamera& cam_left, const PinHoleCamera& cam_right,
            const Isometry3<fptype>& RT)
            :cam_left(cam_left), cam_right(cam_right), RT(RT){
    }

private:
    const PinHoleCamera cam_left;
    const PinHoleCamera cam_right;
    const Isometry3<fptype> RT;
    const Eigen::Matrix<fptype, 3, 4> P = toKMat<fptype>(cam_right) * RT.matrix().template topRows<3>();

    array_type robust_error_sqrNorm_acc = array_type(0);
public:
    template<bool useHuber, bool useWeights, bool is_full>
    void calculate_error(
            const Eigen::Matrix<array_type, 3, 1>& pts3d,
            const Eigen::Matrix<array_type, 2, 1>& left,
            const Eigen::Matrix<array_type, 2, 1>& right,
			const array_type* pHuberL,
			const array_type* pHuberR,
			const array_type* pWeightLeft,
			const array_type* pWeightRight,
            unsigned count = array_type::size
    ){
        Eigen::Matrix<array_type, 2, 1> error_left;
        Eigen::Matrix<array_type, 2, 1> error_right;
        Eigen::Matrix<array_type, 2, 1> rho_left;
        Eigen::Matrix<array_type, 2, 1> rho_right;
        calculate_error<useHuber, useWeights, is_full>(
                pts3d, left, right, pHuberL, pHuberR, pWeightLeft, pWeightRight,
				error_left, error_right,
                useHuber ? &rho_left : nullptr,
                useHuber ? &rho_right : nullptr,
                count);
		assert(check_all_finite(error_left, count));
		assert(check_all_finite(error_right, count));
        if(!is_full && count != array_type::size){
            assert(check_zero_padding(error_left, count));
            assert(check_zero_padding(error_right, count));
            assert(error_left[0].data.allFinite());
            assert(error_left[1].data.allFinite());
            assert(error_right[0].data.allFinite());
            assert(error_right[1].data.allFinite());
            if(useHuber) {
                assert(check_zero_padding(rho_left, count));
                assert(check_zero_padding(rho_right, count));
                assert(rho_left[0].data.allFinite());
                assert(rho_left[1].data.allFinite());
                assert(rho_right[0].data.allFinite());
                assert(rho_right[1].data.allFinite());
            }
        }
    }

    template<bool useHuber, bool useWeights, bool is_full>
    void calculate_error(
            const Eigen::Matrix<array_type, 3, 1> &pts3d,
            const Eigen::Matrix<array_type, 2, 1> &left,
            const Eigen::Matrix<array_type, 2, 1> &right,
			const array_type* pHuberL,
			const array_type* pHuberR,
			const array_type* pWeightLeft,
			const array_type* pWeightRight,
            Eigen::Matrix<array_type, 2, 1> &error_left,
            Eigen::Matrix<array_type, 2, 1> &error_right,
            Eigen::Matrix<array_type, 2, 1> *rho_left = nullptr,
            Eigen::Matrix<array_type, 2, 1> *rho_right = nullptr,
            unsigned count = array_type::size
    ){
        assert((!is_full && count <= array_type::size)
               || (is_full && count == array_type::size));
        assert(std::abs(RT.translation().norm() - 1) < 1E-3f);
        const bool need_padding = (!is_full && count != array_type::size);
        {//left
            const array_type z_inv(1 / pts3d[2].data);
            error_left << pts3d[0] * cam_left.f.x * z_inv + cam_left.c.x - left[0],
                    pts3d[1] * cam_left.f.y * z_inv + cam_left.c.y - left[1];
            if(useHuber){
				const array_type::data_type& huber_delta = pHuberL->data;
				const array_type::data_type huber_delta_x2 = huber_delta * 2;
				const array_type::data_type sqr_huber_delta = square(huber_delta);
                const array_type::data_type e = error_left.squaredNorm().data;
                const Eigen::Array<bool, array_type::size, 1> mask_left = (e < sqr_huber_delta).eval();
				if(!mask_left.all()) {
                    const array_type::data_type sqrte_left = e.cwiseSqrt();
                    (*rho_left)[0].data = mask_left.select(e, huber_delta_x2 * sqrte_left - sqr_huber_delta);
                    (*rho_left)[1].data = mask_left.select(array_type::data_type::Ones(), huber_delta / sqrte_left);
                    assert(check_all_finite((*rho_left), count));
                }
				else{
                    (*rho_left)[0].data = e;
                    (*rho_left)[1].data.setConstant(1);
                }
            }
            if(need_padding){
                set_zero_padding(error_left, count);
                set_zero_padding(*rho_left, count);
            }
        }
        {//right
            const array_type x = pts3d[0] * P(0,0) + pts3d[1] * P(0,1) + pts3d[2] * P(0,2) + P(0, 3);
            const array_type y = pts3d[0] * P(1,0) + pts3d[1] * P(1,1) + pts3d[2] * P(1,2) + P(1, 3);
            const array_type z = pts3d[0] * P(2,0) + pts3d[1] * P(2,1) + pts3d[2] * P(2,2) + P(2, 3);
            const array_type z_inv = fptype(1) / z;
            error_right << x * z_inv - right[0], y * z_inv - right[1];
            if(useHuber){
				const array_type::data_type& huber_delta = pHuberR->data;
				const array_type::data_type huber_delta_x2 = huber_delta * 2;
				const array_type::data_type sqr_huber_delta = square(huber_delta);
                const array_type::data_type e = error_right.squaredNorm().data;
                const Eigen::Array<bool, array_type::size, 1> mask_right = (e < sqr_huber_delta).eval();
				if(!mask_right.all()) {
                    const array_type::data_type sqrte_right = e.cwiseSqrt();
                    (*rho_right)[0].data = mask_right.select(e, huber_delta_x2 * sqrte_right - sqr_huber_delta);
                    (*rho_right)[1].data = mask_right.select(array_type::data_type::Ones(), huber_delta / sqrte_right);
                    assert(check_all_finite((*rho_right), count));
                }else{
                    (*rho_right)[0].data = e;
                    (*rho_right)[1].data.setConstant(1);
                }
            }
            if(need_padding){
                set_zero_padding(error_right, count);
                set_zero_padding(*rho_right, count);
            }
        }
        {// error
            if(useHuber){
				const array_type robustWeightLeft = useWeights ? *pWeightLeft * (*rho_left)[1] : (*rho_left)[1];
				const array_type robustWeightRight = useWeights ? *pWeightRight * (*rho_right)[1] : (*rho_right)[1];
                robust_error_sqrNorm_acc += (*rho_left)[0] * robustWeightLeft;
                robust_error_sqrNorm_acc += (*rho_right)[0] * robustWeightRight;
            }
			else {
				if (useWeights) {
					robust_error_sqrNorm_acc += error_left.squaredNorm() * *pWeightLeft;
					robust_error_sqrNorm_acc += error_right.squaredNorm() * *pWeightRight;
				}
				else {
					robust_error_sqrNorm_acc += error_left.squaredNorm();
					robust_error_sqrNorm_acc += error_right.squaredNorm();
				}
            }
        }
    }
    //get accumulated robust error norm
    void get_E(fptype &error_norm_robust) const{
        error_norm_robust = std::sqrt(robust_error_sqrNorm_acc.sum());
    }
};

} // namespace rsfm::legacy
#pragma GCC diagnostic pop
