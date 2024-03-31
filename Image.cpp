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

#include "Image.h"
#include "Builder.h"
#include <Runtime.hpp>
#include "Config.h"
#include "FiberUtils.h"
#include <RapidSift.h>
#include <CudaMemPool.h>
#include "Types.h"
#include <macros.h>
#include <vector>
#include "SfmUtils.h"
#include <checksum.h>
#include <fstream> // for debug
#include <ImageReader.h>

namespace rsfm{

// useGlCoords defines coordinate style of pt. true means OpenGL/CUDA/D3D texture style and false means opencv style
// returns color in rgb order
template<bool useGlCoords = true>
inline uchar3 sampleImage(const cudapp::Image8UC3 &img, float2 pt)
{
    assert(!img.empty());
    if (useGlCoords){
        pt.x -= 0.5f;
        pt.y -= 0.5f;
    }

    int x[2];
    int y[2];
    for(int i = 0; i < 2; i++)
    {
        x[i] = std::clamp(static_cast<int>(std::floor(pt.x)) + i, 0, int(img.width() - 1));
        y[i] = std::clamp(static_cast<int>(std::floor(pt.y)) + i, 0, int(img.height() - 1));
    }

	using Pixel = cudapp::Image8UC3::Pixel;
    Pixel corners[2][2];

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
            corners[i][j] = img(y[j], x[i]);
#pragma GCC diagnostic pop
        }
    }

    float x0 = pt.x - x[0], x1 = x[1] - pt.x;
    float y0 = pt.y - y[0], y1 = y[1] - pt.y;
    float scale = 1.f / ((x0 + x1) * (y0 + y1));
    float weight[2][2] = {
            {x1 * y1 * scale, x1 * y0 * scale},
            {x0 * y1 * scale, x0 * y0 * scale}
    };

    Pixel rgb;
    for(int c = 0; c < 3; c++)
    {
        rgb[c] = static_cast<uint8_t>(std::clamp(static_cast<int>(round(corners[0][0][c] * weight[0][0]
                                                                    + corners[0][1][c] * weight[0][1]
                                                                    + corners[1][0][c] * weight[1][0]
                                                                    + corners[1][1][c] * weight[1][1])), 0, 255));
    }

    return {rgb[0], rgb[1], rgb[2]};
}

Image createImage(Builder* builder, fs::path file, ImageHandle hImage, CameraHandle hCamera, PoseHandle hPose, const std::vector<TiePtMeasurement>& tiePtMeasurements, RapidSift* detector)
{
    if (!fs::exists(file)){
        fprintf(stderr, "[ERROR] File %s is not found\n", file.c_str());
        DIE("File not found");
    }
	auto md5sum = cudapp::md5sum(file.c_str());
	auto rgbImg = [&file](){
		auto reader = cudapp::createImageReader();
		reader->setFile(file.c_str());
		return std::make_shared<const cudapp::Image8UC3>(reader->decodeTo8UC3());
	}();
    const int width = rgbImg->width();
    const int height = rgbImg->height();
    const auto resolution = builder->getCamResolution(hCamera);
    if (resolution.x != cast32u(width) || resolution.y != cast32u(height)) {
        fprintf(stderr, "Expecting {%u, %u}, got {%u, %u}\n", resolution.x, resolution.y, cast32u(width), cast32u(height));
    }
    ASSERT(resolution.x == cast32u(width) && resolution.y == cast32u(height));

    const auto grayImg = std::make_shared<const cudapp::Image8U>(rgbToGray(*rgbImg));

    const auto& cfg = builder->config();
    std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>> futureFeatures;
    if (cfg.sift.uniform) {
        futureFeatures = detector->uniform_detect_compute_and_abstract([grayImg](){return grayImg->data();}, width, height, cfg.sift.targetNbKPoints, cfg.sift.minOverDetectRatio, cfg.sift.initContrastThreshold, cfg.sift.upSample, cfg.sift.minContrastThreshold, cfg.bow.nbAbstractSamples);
    }
    else {
        DIE("non-uniform detection does not yet support abstract");
        // futureFeatures = detector->detect_and_describe([grayImg](){return grayImg->data;}, width, height, cfg.sift.initContrastThreshold, cfg.sift.upSample);
    }
#if 0
    if (cfg.sift.uniform) {
        futureFeatures = detector->uniform_detect_compute_and_abstract([grayImg](){return grayImg->data();}, width, height, cfg.sift.targetNbKPoints, cfg.sift.minOverDetectRatio, cfg.sift.initContrastThreshold, cfg.sift.upSample, cfg.sift.minContrastThreshold, cfg.bow.nbAbstractSamples);
    }
    else {
        DIE("non-uniform detection does not yet support abstract");
        // futureFeatures = detector->detect_and_describe([grayImg](){return grayImg->data;}, width, height, cfg.sift.initContrastThreshold, cfg.sift.upSample);
    }
#endif
    std::vector<KeyPoint> kpoints;
    std::vector<SiftDescriptor> descriptors;
    std::vector<bool> abstractMask;
    std::tie(kpoints, descriptors, abstractMask) = builder->fiberBlockingService()->delegate(std::move(futureFeatures)).get();
    const auto nbNormalKPts = static_cast<uint32_t>(kpoints.size()); // @fixme: to be fixed.

    const size_t nbKPts = kpoints.size();
    assert(kpoints.size() == descriptors.size());
    const uint32_t id = static_cast<uint32_t>(hImage);
#if 0
    {
        errno = 0;
        std::ofstream fout(makeFmtStr("desc/dbg_%u.desc", id), std::ios::binary | std::ios::trunc);
        fout.write(reinterpret_cast<const char*>(descriptors.data()), sizeof(SiftDescriptor) * descriptors.size());
        if(!fout.good()) {
            const char * errorStr = strerror(errno);
            printf("Error: %s\n", errorStr);
        }
        fout.close();
    }
#endif
    using cudapp::storage::DiskStoragePolicy;
    const auto kptsKey = builder->registerCacheableData(kpoints, makeFmtStr("%u.kpts", id), DiskStoragePolicy::kImmutable, false);
    const auto descKey = builder->registerCacheableData(descriptors, makeFmtStr("%u.desc", id), DiskStoragePolicy::kImmutable, false);

	std::vector<TiePtMeasurementExt> tiePts(tiePtMeasurements.size());
	std::transform(tiePtMeasurements.begin(), tiePtMeasurements.end(), tiePts.begin(), [rgbImg](const TiePtMeasurement& in){
		const uchar3 color = sampleImage<true>(*rgbImg, float2{in.x, in.y});
		return TiePtMeasurementExt{.hTiePt = in.hTiePt, .x = in.x, .y = in.y, .color = color};
	});

    const cudaStream_t stream = builder->anyStream();
    auto kptsColor = builder->cudaMemPool<CudaMemType::kSystem>().alloc<uchar3>(kpoints.size(), stream);
    assert(kptsColor.size() == nbKPts);
    launchCudaHostFunc(stream, [kpoints{std::move(kpoints)}, rgbImg, kptsColorPtr{kptsColor.get()}]{
        auto colorPicker = [&rgbImg](const KeyPoint& kpoint){
            return sampleImage<true>(*rgbImg, kpoint.location);
        };
        std::transform(kpoints.begin(), kpoints.end(), kptsColorPtr, colorPicker);
    });
    const auto kptsColorKey = builder->registerCacheableData(std::move(kptsColor), makeFmtStr("%u.color", id), DiskStoragePolicy::kImmutable, false);

    return Image{hImage, hCamera, hPose, std::move(tiePts), file, md5sum, width, height, static_cast<uint32_t>(nbKPts), nbNormalKPts, kptsKey, descKey, kptsColorKey,  mask2indices(abstractMask)};
}

} // namespace rsfm
