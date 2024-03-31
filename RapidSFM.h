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
#include <type_traits>
#include <limits>
#include <cstdint>
#include <cmath>

namespace rsfm
{
template <typename T>
constexpr std::enable_if_t<std::is_enum<T>::value, T> invalidVal() {
    return static_cast<T>(std::is_unsigned<std::underlying_type_t<T>>::value ?
        std::numeric_limits<std::underlying_type_t<T>>::max() :
        static_cast<std::underlying_type_t<T>>(-1));
}

template <typename T>
constexpr std::enable_if_t<std::is_integral<T>::value, T> invalidVal() {
    return static_cast<T>(std::is_unsigned<T>::value ?
        std::numeric_limits<T>::max() : static_cast<T>(-1));
}

#if __cplusplus >= 201703L
template <typename T>
inline constexpr T kInvalid = invalidVal<T>();
#endif

enum class PointHandle : uint32_t{};
enum class CameraHandle : uint32_t{};
enum class PoseHandle : uint32_t{};
enum class ImageHandle : uint32_t{};
enum class TiePtHandle : uint32_t{};

enum class IntriType : uint32_t
{
    kF1,        // f
    kF2,        // fx, fy
    kF2C2,      // f, cx, cy
    kF1D2,      // f, k1, k2
    kF1D5,      // f, k1, k2, p1, p2, k3
    kF1C2D5,    // f, cx, cy, k1, k2, p1, p2, k3
    kF2D5,      // fx, fy, k1, k2, p1, p2, k3
    kF2C2D5     // fx, fy, cx, cy, k1, k2, p1, p2, k3
};

class IModel
{
public:
    virtual ~IModel();
    virtual void writePly(const char* filename) const = 0;
    virtual void writeNvm(const char* filename) const = 0;
    virtual void writeRsm(const char* filename) const = 0;
};

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = INFINITY;
constexpr size_t kMaxObliqueCameraBundleSize = 1u;

struct TiePtMeasurement
{
    TiePtHandle hTiePt;
    float x;
    float y;
};

struct Covariance3{
    float xx, yy, zz;
    float xy, xz, yz;

    //! inf means zero information matrix and no control
    static Covariance3 inf() {
        return {kInf, kInf, kInf, 0, 0, 0};
    }
    //! fixed means infinity information matrix and fixed control
    static Covariance3 fixed() {
        return {0, 0, 0, 0, 0, 0};
    }
    bool isInf() const {
        return std::isinf(xx) && std::isinf(yy) && std::isinf(zz) && xy == 0 && xz == 0 && yz == 0;
    }
    bool isFixed() const {
        return xx == 0 && yy == 0 && zz == 0 && xy == 0 && xz == 0 && yz == 0;
    }
};

class IBuilder
{
public:
    enum PipelineType{
        kIncremental
    };

    virtual ~IBuilder();
    virtual PipelineType getPipelineType() const = 0;
    virtual size_t getNbPipelineStages() const = 0;
    virtual IntriType getIntriType() const = 0;
    // For now this is always 1, i.e. monocular cameras.
    virtual size_t getObliqueCameraBundleSize() const = 0;

    using ProgressCallback = void(*)(void*);
    virtual void setProgressCallback(ProgressCallback callback, void* data) = 0;

    virtual TiePtHandle addTiePoint() = 0;
    // A control point is a tie point with known fixed 3D position.
    virtual TiePtHandle addControlPoint(double x, double y, double z, const Covariance3& cov = Covariance3::fixed(), float huber = INFINITY) = 0;
    // if fy == kNaN, it is initialized to fx. If cx/cy is kNaN, it is initialized to center of the images
    virtual CameraHandle addCamera(uint32_t resX, uint32_t resY, float fx, float fy = kNaN, float cx = kNaN, float cy = kNaN, float k1 = 0.f, float k2 = 0.f, float p1 = 0.f, float p2 = 0.f, float k3 = 0.f) = 0;
    // RVec = {rx,ry,rz}, C = {cx, cy, cz}, cov is for C. RVec is not used so far. v = {vx,vy,vz} is for rolling shutter. Should be the movement between two consecutive CMOS row reading. If v is specified to be non-zero, r must be valid. v is not used so far.
    virtual PoseHandle addPose(float rx = kNaN, float ry = kNaN, float rz = kNaN, double cx = kNaN, double cy = kNaN, double cz = kNaN, const Covariance3& cov = Covariance3::inf(), float huber = kInf, float vx = 0.f, float vy = 0.f, float vz = 0.f) = 0;
    // if pose is set to kInvalid, a new pose dedicated for this image is created.
    virtual ImageHandle addImage(const char* file, CameraHandle camera, PoseHandle pose = invalidVal<PoseHandle>(),
        const TiePtMeasurement* tiePtMeasurements = nullptr, size_t nbTiePtMeasurements = 0) = 0;


    // nbPendingTasks length should be getNbPipelineStages() + 1
    virtual void getPipelineStatus(size_t* nbPendingTasks) const = 0;

    virtual void finish() = 0;
    virtual size_t getNbModels() const = 0;
    virtual const IModel* getModel(size_t idx) const = 0;

    enum {
        kPLY = 1,
        kNVM = 1 << 1,
        kRSM = 1 << 2
    };
    virtual void setSavedName(ImageHandle hImage, const char* name) = 0; // name to use when saving results (e.g. NVM and RSM)
    virtual void writeClouds(const char* filenamePrefix = "cloud_", uint32_t flag = kPLY) const = 0;
};

IBuilder* createBuilder();
IBuilder* createBuilder(const char* yamlDoc);
IBuilder* createBuilder(const char* yamlDocs[], size_t nbDocs);

} // namespace rsfm
