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

#include <RapidBA.h>
#include "Types.hpp"

namespace rsfm
{
inline const Vec3f Pose::transform(const Vec3f& p0, rba::ShutterType shutter, float y, float rollingCenter) const {
	switch (shutter) {
	case rba::ShutterType::kGlobal: return R * (p0 - C);
	case rba::ShutterType::kRolling1D:
	case rba::ShutterType::kRolling3D:
	case rba::ShutterType::kRollingFixedVelocity: return R * (p0 - C - v * (y - rollingCenter)); break;
	case rba::ShutterType::kRolling1DLoc: return R * (p0 - C) + v * (y - rollingCenter); break;
	}
	throw std::runtime_error("you should never reach here");
}
}
