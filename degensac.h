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

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
namespace rsfm
{
struct DegensacSolution
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix3f F;
    int32_t nbInliersF;
    std::vector<bool> inlierMaskF;

    Eigen::Projective2f H; // invalid if nbInliersH is zero
    int32_t nbInliersH;
    std::vector<bool> inlierMaskH;
};

DegensacSolution findEpipolarityDegensac(
        const std::array<Eigen::ArrayXf, 4>& ptPairs, const std::array<Eigen::ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim,
        float thresRelativeF, float thresRelativeH,
        float requiredRansacConfidence);
}

