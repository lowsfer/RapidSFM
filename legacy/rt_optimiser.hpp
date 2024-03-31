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

#include <eigen3/Eigen/Dense>
#include <cmath>
#include "aligned_array.hpp"
#include "../Types.hpp"
#include "../SfmUtils.hpp"

namespace rsfm::legacy
{

#pragma GCC diagnostic push
#ifdef NDEBUG
#pragma GCC optimize("-ffast-math", "-fno-finite-math-only")
#endif
class rt_optimiser
{
public:
    using fptype = float;
	static constexpr Eigen::Index vecSize = alignof(Eigen::Matrix<float, 32, 1>) / sizeof(fptype);
    using array_type = aligned_array<fptype, vecSize>;
#ifdef NDEBUG
    static_assert(alignof(array_type) == 32); // AVX256
#endif
    enum class enum_state{
        OK, NotConverged, NumericFailure
    };
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	rt_optimiser();

    void set_camera(const PinHoleCamera& cam0, const PinHoleCamera& cam1){
        cam_left = cam0;
        cam_right = cam1;
    }
    void set_input(const Isometry3<fptype> &RT_init, const Eigen::Matrix<fptype, -1, 4> &point_pairs,
                   const Eigen::Matrix<fptype, -1, 3> *points3d_init = nullptr);

	// we use uniform huber / weight for errors in both image x and y dimension
	enum_state optimise(bool verbose = false, const Eigen::VectorXf& huberLeft = {}, const Eigen::VectorXf& huberRight = {}, const Eigen::VectorXf weightsLeft = {}, const Eigen::VectorXf weightsRight = {});

	Isometry3<fptype> get_RT() const {return RT;}
	Eigen::Matrix<fptype, -1, 3> get_points() const;
	uint32_t getNbPts() const {
		return num_points;
	}
	std::vector<bool> get_inlier_mask(const fptype* sqrThresLeft, const fptype* sqrThresRight) const;
private:
    template<bool use_huber, bool useWeights>
    enum_state optimiseImpl(bool verbose, const Eigen::VectorXf& huberLeft, const Eigen::VectorXf& huberRight, const Eigen::VectorXf weightsLeft, const Eigen::VectorXf weightsRight);

    void set_num_points(unsigned n);

    //calculates error, jacobian and g_a, g_b, U, V, W, g and error_norm_robust
    template<bool use_huber, bool useWeights>
    void calculate_error_jacobian_UVWgE(const EigenAlignedVector<array_type>& huberLeft, const EigenAlignedVector<array_type>& huberRight, const EigenAlignedVector<array_type>& weightsLeft, const EigenAlignedVector<array_type>& weightsRight);

	bool check_stop_condition1() const;
	bool check_stop_condition2(fptype delta_norm) const;
	bool check_stop_condition3() const;
	bool check_stop_condition4(fptype error_norm_new) const;

    //if pts3d.z becomes large, single-float precision will cause failure. We stop in such case.
    bool check_z_range() const;//in (0, 100.f) range

    PinHoleCamera cam_left;
    PinHoleCamera cam_right;
    EigenAlignedVector<Eigen::Matrix<array_type, 2, 1>> left;
    EigenAlignedVector<Eigen::Matrix<array_type, 2, 1>> right;
    std::array<fptype, 4> epsilons = {{1E-4f, 1E-4f, 1E-4f, 0}};
	int max_iters = 60;

	int iter;
	fptype nu;
	fptype lambda;//mu

	int RT_update_counter = 0;
	void orthogonalise(){
		if(RT_update_counter >= 32) {
            RT.linear() = orthogonalised(RT.linear());
            RT.translation().normalize();
			RT_update_counter = 0;
		}
	}

    const Isometry3<fptype> updated_RT(const Eigen::Matrix<fptype, 5, 1>& delta) const{
        Isometry3<fptype> result = Isometry3<fptype>::Identity();
        result.linear() = RT.linear() * gvec2mat(&delta[0]);
        const Vector2<fptype> Tgvec = UnitT2Tgvec(RT.translation()).template topRows<2>();
        //optimised version of gvec_mul() function with z==0 for both gvec's
        auto gvec_mul = [](const Vector2<fptype>& b, const Vector2<fptype>& a){
//            return (a + b + b.cross(a)) / (1 - a.dot(b));//for Vector3
            Vector3<fptype> result;
            result << a + b, a[1] * b[0] - a[0] * b[1];
            result *= 1 / (1 - a.dot(b));
            return result;
        };
        const Vector3<fptype> Tgvec_new = gvec_mul(Tgvec, Vector2<fptype>::Map(&delta[3]));
        result.translation() = gvec2mat(Tgvec_new).template rightCols<1>();
        return result;
    }

	Isometry3<fptype> RT;
    EigenAlignedVector<Eigen::Matrix<array_type, 3, 1>> pts3d;
    fptype pts3d_norm;
#ifndef NDEBUG
    EigenAlignedVector<Eigen::Matrix<array_type, 3, 1>> pts3d_orig;
#endif


//    eigen_aligned_vector<Eigen::Matrix<array_type, 2, 1>> error_left;
//    eigen_aligned_vector<Eigen::Matrix<array_type, 2, 1>> error_right;
//    eigen_aligned_vector<Eigen::Matrix<array_type, 2, 3>> jacobian_left_pt;
//    eigen_aligned_vector<Eigen::Matrix<array_type, 2, 5>> jacobian_right_cam;
//    eigen_aligned_vector<Eigen::Matrix<array_type, 2, 3>> jacobian_right_pt;
//    eigen_aligned_vector<array_type> rho_left;//Huber robustifying factor

	Eigen::Matrix<fptype, 5, 1> g_a;
    EigenAlignedVector<Eigen::Matrix<array_type, 3, 1>> g_b;
	Eigen::Matrix<fptype, 5, 5> U;
    EigenAlignedVector<Eigen::Matrix<array_type, 3, 3>> V;
    EigenAlignedVector<Eigen::Matrix<array_type, 5, 3>> W;
    fptype error_norm_robust;
    fptype g_absmax;

    unsigned num_points;
    unsigned num_blocks;
    unsigned num_whole_blocks;
    unsigned num_residue_points;

    //set as class member to avoid dynamic allocation
    EigenAlignedVector<Eigen::Matrix<array_type, 3, 1>> delta_b;
};

#pragma GCC diagnostic pop

} // namespace rsfm::legacy
