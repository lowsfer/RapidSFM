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
#include <cstddef>
#include "Types.h"

namespace rsfm
{
template <typename T, uint32_t nbParams, uint32_t nbIters = 2> // typically 2 iterations is enough
Vec2<T> undistort(const Vec2<T>& __restrict__ distortedNormXY, const std::array<float, nbParams>& __restrict__ params);
// px, py are the 2d image coordinates in pixels
void undistortInPlace(float px[], float py[], size_t nbPts, const RealCamera& camera);
} // namespace rsfm