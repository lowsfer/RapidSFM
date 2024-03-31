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

#include "rt_optimiser.hpp"
#include "rt_optimiser_helper.hpp"
#include "geometry.hpp"
#include <cpp_utils.h>

namespace rsfm::legacy
{
rt_optimiser::rt_optimiser()
{
}

rt_optimiser::enum_state rt_optimiser::optimise(bool verbose, const Eigen::VectorXf& huberLeft, const Eigen::VectorXf& huberRight, const Eigen::VectorXf weightsLeft, const Eigen::VectorXf weightsRight)
{
	auto checker = [this](const Eigen::VectorXf& l, const Eigen::VectorXf& r) {
		ASSERT(l.size() == 0 || l.size() == num_points);
		ASSERT(r.size() == 0 || r.size() == num_points);
		ASSERT(l.size() == r.size());
		return l.size() != 0;
	};
	const bool useHuber = checker(huberLeft, huberRight);
	const bool useWeights = checker(weightsLeft, weightsRight);
	if (useHuber) {
		if (useWeights) {
			return optimiseImpl<true, true>(verbose, huberLeft, huberRight, weightsLeft, weightsRight);
		}
		else {
			return optimiseImpl<true, false>(verbose, huberLeft, huberRight, weightsLeft, weightsRight);
		}
	}
	else {
		if (useWeights) {
			return optimiseImpl<false, true>(verbose, huberLeft, huberRight, weightsLeft, weightsRight);
		}
		else {
			return optimiseImpl<false, false>(verbose, huberLeft, huberRight, weightsLeft, weightsRight);
		}
	}
}

//@fixme: use DCS instead of Huber
template<bool use_huber, bool useWeights>
rt_optimiser::enum_state rt_optimiser::optimiseImpl(bool verbose, const Eigen::VectorXf& huberLeft_, const Eigen::VectorXf& huberRight_, const Eigen::VectorXf weightsLeft_, const Eigen::VectorXf weightsRight_)
{
	auto toBlocks = [](const Eigen::VectorXf& src) {
		EigenAlignedVector<array_type> dst(divUp(src.rows(), vecSize));
		const auto nbFullBlocks = src.rows() / vecSize;
		for (Eigen::Index i = 0; i < nbFullBlocks; i++) {
			dst.at(i).data = src.template middleRows<vecSize>(vecSize * i);
		}
		const auto residueSize = src.rows() % vecSize;
		if (residueSize != 0) {
			for (Eigen::Index i = 0; i < vecSize; i++) {
				dst.at(nbFullBlocks).data[i] = i < residueSize ? src[vecSize * nbFullBlocks + i] : 0.f;
			}
		}
		return dst;
	};
	const EigenAlignedVector<array_type> huberLeft = toBlocks(huberLeft_);
	const EigenAlignedVector<array_type> huberRight = toBlocks(huberRight_);
	const EigenAlignedVector<array_type> weightsLeft = toBlocks(weightsLeft_);
	const EigenAlignedVector<array_type> weightsRight = toBlocks(weightsRight_);
    enum_state state = enum_state::OK;
    iter = 0;
    nu = 2;
    calculate_error_jacobian_UVWgE<use_huber, useWeights>(huberLeft, huberRight, weightsLeft, weightsRight);
    bool stop = check_stop_condition1();
    if (stop) {
        printf("[RT Optimizer] No need to optimise RT (g_absmax = %e). This is too good to be true and may be a bug.\n", g_absmax);
    }
    constexpr fptype tao = 1E-6f;
    lambda = tao * max(
            U.diagonal().maxCoeff(),
            array_mat_max(V, 0, 0, num_points),
            array_mat_max(V, 1, 1, num_points),
            array_mat_max(V, 2, 2, num_points));
    lambda = clamp(lambda, 1E-6f, 1E-2f);

    while(!stop && iter < max_iters)
    {
        if(!check_z_range()){
            state = enum_state::NumericFailure;
            if (verbose) {
                printf("[RT Optimizer] numeric failure\n");
            }
            break;
        }

        if(verbose)
            printf("[RT Optimizer] Iteration #%d: %f (lambda = %e, g_absmax = %f)\n", iter, error_norm_robust, lambda, g_absmax);
        iter++;
        fptype rho = 0;
        do{
            const Eigen::Matrix<fptype, 5, 5> U_damp = U + lambda * Eigen::Matrix<fptype, 5, 5>::Identity();

            assert(check_array_mat_zero_padding(W, num_points));
            assert(check_array_mat_zero_padding(V, num_points));
            assert(check_array_mat_zero_padding(g_b, num_points));

            Eigen::Matrix<array_type, 5, 5> YWt_acc;
            YWt_acc.setConstant(array_type(0));
            Eigen::Matrix<array_type, 5, 1> YEb_acc;
            YEb_acc.setConstant(array_type(0));
            for (unsigned n = 0; n < num_blocks; n++) {
                const Eigen::Matrix<array_type, 3, 3> V_damp = [&]() {
                    Eigen::Matrix<array_type, 3, 3> result = V[n];
                    for (int i = 0; i < 3; i++) {
                        result(i, i) += lambda;
                    }
                    return result;
                }();
#if 0
                assert(check_all_finite(W[n]));
                const Eigen::Matrix<array_type, 3, 3> V_damp_inv = V_damp.inverse();//done: implement LU or gauss elimination instead of inverse!
                assert(check_all_finite(V_damp_inv));
                const Eigen::Matrix<array_type, 5, 3> Y = W[n] * V_damp_inv;
                assert(check_all_finite(Y));
                YWt_acc += Y * W[n].transpose();
                assert(check_all_finite(g_b[n]));
                YEb_acc += Y * g_b[n];
#else
                Eigen::Matrix<array_type, 3, 6> Wt_gb;
                Wt_gb << W[n].transpose() , g_b[n];
                assert(check_all_finite(Wt_gb));
                // V_damp may be ill-conditioned and generate NaN here.
                const Eigen::Matrix<array_type, 5, 6> delta_YWt_WEb = W[n] * solveGaussElim(V_damp, Wt_gb);
                // assert(check_all_finite(delta_YWt_WEb));
                YWt_acc += delta_YWt_WEb.template leftCols<5>();
                YEb_acc += delta_YWt_WEb.template rightCols<1>();
#endif
            }
            const Eigen::Matrix<fptype, 5, 5> S = U_damp - array_mat_sum(YWt_acc);//A
            const Eigen::Matrix<fptype, 5, 1> b_a = g_a - array_mat_sum(YEb_acc);//b
            const Eigen::Matrix<fptype, 5, 1> delta_a = S.llt().solve(b_a);
            if (!delta_a.allFinite()) {
                if (verbose) {
                    printf("[RT Optimizer] numeric failure\n");
                }
                return enum_state::NumericFailure;
            }

            array_type delta_b_sqrNorm_acc = array_type(0);
            for (unsigned n = 0; n < num_blocks; n++) {
                const Eigen::Matrix<array_type, 3, 1> b_b = g_b[n] - [&]() {
                    Eigen::Matrix<array_type, 3, 1> result;
                    for (int i = 0; i < 3; i++) {
                        result[i] = W[n](0, i) * delta_a[0] + W[n](1, i) * delta_a[1] + W[n](2, i) * delta_a[2]
                                    + W[n](3, i) * delta_a[3] + W[n](4, i) * delta_a[4];
                    }
                    return result;
                }(); //g_b - W.transpose() * delta_a;
                const Eigen::Matrix<array_type, 3, 3> V_damp = [&]() {
                    Eigen::Matrix<array_type, 3, 3> result = V[n];
                    for (int i = 0; i < 3; i++) {
                        result(i, i) += lambda;
                    }
                    return result;
                }();
//			const Eigen::Matrix<array_type, 3, 1> delta_b = V_damp.llt().solve(b_b);
//                delta_b[n] = V_damp.inverse() * b_b;
                delta_b[n] = solveGaussElim(V_damp, b_b);
                assert(check_all_finite(delta_b[n]));
                for(int i = 0; i < delta_b[n].size(); i++)
                    delta_b_sqrNorm_acc += square(delta_b[n][i]);
            }
            const fptype delta_norm = std::sqrt(delta_a.squaredNorm() + delta_b_sqrNorm_acc.sum());
            if(check_stop_condition2(delta_norm)){
                stop = true;
            }
            else{
                const Isometry3<fptype> RT_new = updated_RT(-delta_a);
                //calculate rho
                const fptype error_norm_new = [&]()
                {
                    error_calculator calculator(cam_left, cam_right, RT_new);
                    for (unsigned n = 0; n < num_whole_blocks; n++) {
                        const Eigen::Matrix<array_type, 3, 1> pts3d_new = pts3d[n] - delta_b[n];
                        calculator.template calculate_error<use_huber, useWeights, true>(pts3d_new, left[n], right[n],
							use_huber ? &huberLeft[n] : nullptr,
							use_huber ? &huberRight[n] : nullptr,
							useWeights ? &weightsLeft[n] : nullptr,
							useWeights ? &weightsRight[n] : nullptr);
                    }
                    if (num_residue_points != 0) {
                        const unsigned n = num_whole_blocks;
                        const Eigen::Matrix<array_type, 3, 1> pts3d_new = pts3d[n] - delta_b[n];
                        calculator.template calculate_error<use_huber, useWeights, false>(pts3d_new, left[n], right[n],
							use_huber ? &huberLeft[n] : nullptr,
							use_huber ? &huberRight[n] : nullptr,
							useWeights ? &weightsLeft[n] : nullptr,
							useWeights ? &weightsRight[n] : nullptr,
							num_residue_points);
                    }
                    fptype error_norm_new;
                    calculator.get_E(error_norm_new);
                    return error_norm_new;
                }();
                assert(check_array_mat_zero_padding(delta_b, num_points));
                assert(check_array_mat_zero_padding(g_b, num_points));
                const fptype rho_denorminator = [&](){
                    fptype result = delta_a.dot(lambda * delta_a + g_a);
                    array_type acc_b(0);
                    for(unsigned n = 0; n < num_blocks; n++){
                        for(unsigned i = 0; i < 3; i++) {
                            acc_b += delta_b[n][i] * (lambda * delta_b[n][i] + g_b[n][i]);
                        }
                    }
                    result += acc_b.sum();
                    return result;
                }();
                rho = (square(error_norm_robust) - square(error_norm_new)) / rho_denorminator;
                if(verbose)
                    printf("[RT Optimizer]\tsub-iter: %f, %s (lambda = %e, g_absmax = %f)\n", error_norm_new, rho > 0 ? "accept" : "reject", lambda, g_absmax);
				if (rho == 0 && error_norm_new == error_norm_robust && lambda > 1E12f) {
					stop = true; // does not make sense to continue. Will almost always result in lambda -> +inf
					break;
				}
                if (rho > 0 && error_norm_new < error_norm_robust)
                {
                    stop = check_stop_condition4(error_norm_new);
                    RT = RT_new;
                    orthogonalise();
                    {//equivalent to: pts3d += delta_b; calculate_error_jacobian_UVWgE();
                        array_type pts3d_sqrNorm_acc = array_type(0);
                        error_jacobian_calculator calculator(cam_left, cam_right, RT);
                        for(unsigned n = 0; n < num_whole_blocks; n++){
                            pts3d[n] -= delta_b[n];
                            calculator.template calculate_error_jacobian<use_huber, useWeights, true>(
                                    pts3d[n], left[n], right[n],
									use_huber ? &huberLeft[n]: nullptr,
									use_huber ? &huberRight[n]: nullptr,
									useWeights ? &weightsLeft.at(n): nullptr,
									useWeights ? &weightsRight.at(n): nullptr,
//                                    error_left[n], error_right[n],
//                                    jacobian_left_pt[n], jacobian_right_cam[n], jacobian_right_pt[n],
                                    V[n], W[n], g_b[n]);
                            pts3d_sqrNorm_acc += pts3d[n].squaredNorm();
                        }
                        if(num_residue_points != 0) {
                            const unsigned n = num_whole_blocks;
                            pts3d[n] -= delta_b[n];
                            calculator.template calculate_error_jacobian<use_huber, useWeights, false>(
                                    pts3d[n], left[n], right[n],
									use_huber ? &huberLeft[n] : nullptr,
									use_huber ? &huberRight[n] : nullptr,
									useWeights ? &weightsLeft.at(n): nullptr,
									useWeights ? &weightsRight.at(n): nullptr,
//                                    error_left[n], error_right[n],
//                                    jacobian_left_pt[n], jacobian_right_cam[n], jacobian_right_pt[n],
                                    V[n], W[n], g_b[n], num_residue_points);
                            pts3d_sqrNorm_acc += pts3d[n].squaredNorm();
                        }
                        calculator.get_UgE(g_a, U, error_norm_robust, g_absmax);
                        pts3d_norm = std::sqrt(pts3d_sqrNorm_acc.sum());
                    }
                    stop = stop || check_stop_condition1();
                    lambda *= std::max(fptype(1.f/3), 1 - cube(2*rho - 1));
                    nu = 2;
                }
                else
                {
                    lambda *= nu;
                    nu *= 2;
                }
				if (std::isinf(lambda) && lambda > 0) { // can't continue.
					stop = true;
					break;
				}
            }
        }while(rho <= 0 && !stop);
        stop = stop || check_stop_condition3();
    }
    if(state == enum_state::OK && iter == max_iters)
        state = enum_state::NotConverged;
    return state;
}

bool rt_optimiser::check_stop_condition1() const {
    const fptype epsilon = epsilons[0];
    return g_absmax <= epsilon;
}

bool rt_optimiser::check_stop_condition2(fptype delta_norm) const {
    const fptype epsilon = epsilons[1];
    return delta_norm <= epsilon * (pts3d_norm + epsilon);
}

bool rt_optimiser::check_stop_condition3() const {
    const fptype epsilon = epsilons[2];
    return error_norm_robust <= epsilon;
}

bool rt_optimiser::check_stop_condition4(fptype error_norm_new) const {
    const fptype epsilon = epsilons[3];
    return error_norm_robust - error_norm_new < epsilon * error_norm_robust;
}

template<bool use_huber, bool useWeights>
void rt_optimiser::calculate_error_jacobian_UVWgE(
	const EigenAlignedVector<array_type>& huberLeft, const EigenAlignedVector<array_type>& huberRight,
	const EigenAlignedVector<array_type>& weightsLeft, const EigenAlignedVector<array_type>& weightsRight)
{
    array_type pts3d_sqrNorm_acc = array_type(0);
    error_jacobian_calculator calculator(cam_left, cam_right, RT);
    for(unsigned n = 0; n < num_whole_blocks; n++){
        calculator.template calculate_error_jacobian<use_huber, useWeights, true>(
                pts3d[n], left[n], right[n],
				use_huber ? &huberLeft[n] : nullptr,
				use_huber ? &huberRight[n] : nullptr,
				useWeights ? &weightsLeft[n] : nullptr,
				useWeights ? &weightsRight[n] : nullptr,
//                error_left[n], error_right[n],
//                jacobian_left_pt[n], jacobian_right_cam[n], jacobian_right_pt[n],
                V[n], W[n], g_b[n]);
        pts3d_sqrNorm_acc += pts3d[n].squaredNorm();
    }
    if(num_residue_points != 0) {
        const unsigned n = num_whole_blocks;
        assert(check_nan_padding(pts3d[n], num_residue_points));
        calculator.template calculate_error_jacobian<use_huber, useWeights, false>(
                pts3d[n], left[n], right[n],
				use_huber ? &huberLeft[n] : nullptr,
				use_huber ? &huberRight[n] : nullptr,
				useWeights ? &weightsLeft[n] : nullptr,
				useWeights ? &weightsRight[n] : nullptr,
//                error_left[n], error_right[n],
//                jacobian_left_pt[n], jacobian_right_cam[n], jacobian_right_pt[n],
                V[n], W[n], g_b[n], num_residue_points);
        pts3d_sqrNorm_acc.data.topRows(num_residue_points) += pts3d[n].squaredNorm().data.topRows(num_residue_points);
    }
    calculator.get_UgE(g_a, U, error_norm_robust, g_absmax);
    assert(std::isfinite(error_norm_robust));
    pts3d_norm = std::sqrt(pts3d_sqrNorm_acc.sum());
}

void rt_optimiser::set_num_points(unsigned n) {
    num_points = n;
    num_whole_blocks = num_points / array_type::size;
    num_residue_points = num_points % array_type::size;
    num_blocks = (num_points + array_type::size - 1) / array_type::size;
    left.resize(num_blocks);
    right.resize(num_blocks);
    pts3d.resize(num_blocks);
//    error_left.resize(num_blocks);
//    error_right.resize(num_blocks);
//    jacobian_left_pt.resize(num_blocks);
//    jacobian_right_cam.resize(num_blocks);
//    jacobian_right_pt.resize(num_blocks);
    g_b.resize(num_blocks);
    V.resize(num_blocks);
    W.resize(num_blocks);
    delta_b.resize(num_blocks);
}

void rt_optimiser::set_input(const Isometry3<fptype> &RT_init, const Eigen::Matrix<fptype, -1, 4> &point_pairs,
                             const Eigen::Matrix<fptype, -1, 3> *points3d_init) {
    RT = RT_init;
    const Eigen::MatrixX3f* ptr_pts3d = nullptr;
    Eigen::MatrixX3f pts3d_triangulated;
    if(points3d_init)
        ptr_pts3d = points3d_init;
    else{
        pts3d_triangulated = triangulate(cam_left, cam_right, RT_init, point_pairs);
        ptr_pts3d = &pts3d_triangulated;
    }
    const Eigen::MatrixX3f& pts3d_init = *ptr_pts3d;
    set_num_points(point_pairs.rows());
    for(unsigned n = 0; n < num_whole_blocks; n++){
        const auto pts2d = point_pairs.template middleRows<array_type::size>(n * array_type::size);
        left[n][0].data = pts2d.col(0);
        left[n][1].data = pts2d.col(1);
        right[n][0].data = pts2d.col(2);
        right[n][1].data = pts2d.col(3);
        const auto pts3d_block = pts3d_init.template middleRows<array_type::size>(n * array_type::size);
        for(unsigned i = 0; i < 3; i++)
            pts3d[n][i].data = pts3d_block.col(i);
    }
    if(num_residue_points != 0){
        const unsigned n = num_whole_blocks;
        const auto pts2d = point_pairs.middleRows(n * array_type::size, num_residue_points);
        left[n][0].data.topRows(num_residue_points) = pts2d.col(0);
        left[n][1].data.topRows(num_residue_points) = pts2d.col(1);
        right[n][0].data.topRows(num_residue_points) = pts2d.col(2);
        right[n][1].data.topRows(num_residue_points) = pts2d.col(3);
        left[n][0].data.bottomRows(array_type::size - num_residue_points).setConstant(NAN);
        left[n][1].data.bottomRows(array_type::size - num_residue_points).setConstant(NAN);
        right[n][0].data.bottomRows(array_type::size - num_residue_points).setConstant(NAN);
        right[n][1].data.bottomRows(array_type::size - num_residue_points).setConstant(NAN);
        const auto pts3d_block = pts3d_init.middleRows(n * array_type::size, num_residue_points);
        for(unsigned i = 0; i < 3; i++) {
            pts3d[n][i].data.topRows(num_residue_points) = pts3d_block.col(i);
            pts3d[n][i].data.bottomRows(array_type::size - num_residue_points).setConstant(NAN);
        }
    }
    // assert(check_z_range());
#ifndef NDEBUG
    pts3d_orig = pts3d;
#endif
}

Eigen::Matrix<rt_optimiser::fptype, -1, 3> rt_optimiser::get_points() const {
    Eigen::Matrix<rt_optimiser::fptype, -1, 3> result(num_points, 3);
    for(unsigned n = 0; n < num_whole_blocks; n++){
        auto pts3d_block = result.template middleRows<array_type::size>(n * array_type::size);
        for(unsigned i = 0; i < 3; i++)
            pts3d_block.col(i) = pts3d[n][i].data;
    }
    if(num_residue_points != 0){
        const unsigned n = num_whole_blocks;
        auto pts3d_block = result.middleRows(n * array_type::size, num_residue_points);
        for(unsigned i = 0; i < 3; i++) {
            pts3d_block.col(i) = pts3d[n][i].data.topRows(num_residue_points);
        }
    }
    return result;
}

std::vector<bool> rt_optimiser::get_inlier_mask(const fptype* sqrThresLeft, const fptype* sqrThresRight) const {
    std::vector<bool> result(num_points);

    constexpr bool use_huber = false;
	constexpr bool useWeights = false;
    error_calculator calculator(cam_left, cam_right, RT);
    for (unsigned n = 0; n < num_whole_blocks; n++) {
        const Eigen::Matrix<array_type, 3, 1>& p = pts3d[n];
        Eigen::Matrix<array_type, 2, 1> error_left;
        Eigen::Matrix<array_type, 2, 1> error_right;
        calculator.template calculate_error<use_huber, useWeights, true>(p, left[n], right[n], nullptr, nullptr, nullptr, nullptr, error_left, error_right);
        const Eigen::Array<bool, array_type::size, 1> mask =
			(error_left.squaredNorm().data < array_type::data_type::ConstMapType(&sqrThresLeft[vecSize * n]))
			&& (error_right.squaredNorm().data < array_type::data_type::ConstMapType(&sqrThresRight[vecSize * n]));
        for(unsigned i = 0; i < array_type::size; i++)
            result[n * array_type::size + i] = mask[i];
    }
    if (num_residue_points != 0) {
        const unsigned n = num_whole_blocks;
        const Eigen::Matrix<array_type, 3, 1>& p = pts3d[n];
        Eigen::Matrix<array_type, 2, 1> error_left;
        Eigen::Matrix<array_type, 2, 1> error_right;
        calculator.template calculate_error<use_huber, useWeights, false>(p, left[n], right[n], nullptr, nullptr, nullptr, nullptr, error_left, error_right);
		array_type::data_type sqrThresLeftVec, sqrThresRightVec;
		static_assert(array_type::data_type::SizeAtCompileTime == vecSize);
		for (uint32_t i = 0; i < vecSize; i++) {
			sqrThresLeftVec[i] = i < num_residue_points ? sqrThresLeft[vecSize * n + i] : 0.f;
			sqrThresRightVec[i] = i < num_residue_points ? sqrThresRight[vecSize * n + i] : 0.f;
		}
        const Eigen::Array<bool, array_type::size, 1> mask =
			(error_left.squaredNorm().data < sqrThresLeftVec) && (error_right.squaredNorm().data < sqrThresRightVec);
        for(unsigned i = 0; i < num_residue_points; i++)
            result[n * array_type::size + i] = mask[i];
    }
    return result;
}

bool rt_optimiser::check_z_range() const {
    constexpr fptype z_max = 1E4f;
    assert(check_array_mat_nan_padding(pts3d, num_points));
    for(unsigned n = 0; n < num_whole_blocks; n++){
        const auto& z = pts3d[n][2].data;
        const bool is_valid = (z.array() > 0.f && z.array() <= z_max).all();
        if(!is_valid)
            return false;
    }
    if(num_residue_points != 0) {
        const unsigned n = num_whole_blocks;
        const auto& z = pts3d[n][2].data;
        auto is_valid = (z.array() > 0.f && z.array() <= z_max).eval();
        is_valid.bottomRows(array_type::size - num_residue_points).setConstant(true);
        if(!is_valid.all())
            return false;
    }
    return true;
}

} // namespace rsfm::legacy
