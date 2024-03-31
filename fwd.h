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
#include <cuda_utils.h>
#include <StorageFwd.h>

class RapidSift;

namespace rsfm
{

using Coord = float;
template <typename T> struct Vec3;
struct RealCamera;
struct InverseRealCamera;
template <bool isInverse> struct PinHoleCameraTemplate;
using PinHoleCamera = PinHoleCameraTemplate<false>;
using InversePinHoleCamera = PinHoleCameraTemplate<true>;
struct LocCtrl;
struct Image;
struct Pose;
struct ImagePair;
struct Matches;

template <typename T> struct Vec2;

using Index = uint32_t;
extern const Index kInvalidIndex;

class Builder;

struct Config;

template <typename Handle, typename = void>
class HandleGenerator;

class IModel;
class IncreModel;
class ModelBuilder;
class PnPOptimizer;

// There is no way to forward declare inheritance, so we forward declare a cast function
const rsfm::IModel* toInterface(const IncreModel* m);

class BruteForceMatcher;

using IRuntime = cudapp::IRuntime;
using Runtime = cudapp::Runtime;

class RansacMatchFilter;

} // namespace rsfm

namespace rba
{
class IModel;
class IUniversalModel;
}

namespace Eigen
{
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
class Matrix;
}

namespace cudapp
{
class PriorityFiberPool;
class FiberBlockingService;
}