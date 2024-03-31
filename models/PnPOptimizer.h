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
#include <eigen3/Eigen/Geometry>
#include "../Types.hpp"
#include <cpp_utils.h>
#include <RapidBA.h>

namespace rsfm
{
// CPU SIMD implementation.
Pose optimizePnP(const PinHoleCamera& cam, const Pose& initPose,
    const Vec3<const float* __restrict__>& pts3d, const Vec2<const float* __restrict__>& pts2d,
    const float* __restrict__ const omega, const float* __restrict__ const huber, const size_t nbPts,
    const int maxNbSteps = 30, const float initLambda = 0.001f, const bool verbose = false);

} // namespace rsfm
