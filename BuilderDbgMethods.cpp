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

#include "Builder.h"
#include "Config.h"
#include <Runtime.hpp>
#include <algorithm>
#include <RapidSift.h>
#include <vector>
#include <cuda_utils.h>
#include <StorageManager.h>
#include <DefaultCacheableObject.h>
#include "Image.h"
#include "ImagePair.h"
#include "Types.hpp"
#include <FiberUtils.h>
#include "legacy/PropagationMatchFilter.h"
#include <boost/fiber/cuda/waitfor.hpp>
#include <FiberUtils.h>
#include <functional>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using cudapp::storage::StorageLocation;
using cudapp::storage::DefaultCacheableObject;
template <typename T>
using AcquiredMemory = cudapp::storage::AcquiredMemory<T>;

namespace rsfm {

namespace
{
//pixels_at_grid_cross_points is for dst. src kpt is always true.
template<bool pixels_at_grid_cross_points>
const cv::KeyPoint keyPoint2cvKeyPoint(const KeyPoint& kpt){
    cv::KeyPoint ret;
    ret.pt = cv::Point2f(kpt.location.x, kpt.location.y);
    if (!pixels_at_grid_cross_points){
        ret.pt.x -= 0.5f;
        ret.pt.y -= 0.5f;
    }
    ret.size = __half2float(kpt.size);
    ret.angle = kpt.angle * 360.f / 256.f;
    ret.octave = kpt.octave;
    ret.response = __half2float(kpt.response);
    ret.class_id = -1;
    return ret;
}
}

void Builder::dbgDrawMatches(const Matches &matches, const char* prefix)
{
    const cudaStream_t stream = anyStream();
    const Image* first = nullptr;
    const Image* second = nullptr;
    {
        std::unique_lock<std::shared_mutex> lk {mLock};
        first = mImages.at(matches.images.first).get();
        second = mImages.at(matches.images.second).get();
    }
    const auto loc = StorageLocation::kSysMem;
    const auto kptsFirstHolder = cudapp::storage::acquireMemory<const KeyPoint>(storageManager(), first->keyPoints, loc, stream, false, true);
    const auto kptsSecondHolder = cudapp::storage::acquireMemory<const KeyPoint>(storageManager(), second->keyPoints, loc, stream, false, true);
    cudapp::fiberSyncCudaStream(stream);

    auto toCvKPts = [](const AcquiredMemory<const KeyPoint>& holder, size_t nbKPts){
        const KeyPoint* kpts = holder.data();
        REQUIRE(nbKPts == holder.nbElems());
        std::vector<cv::KeyPoint> cvKPts(nbKPts);
        std::transform(kpts, kpts+nbKPts, cvKPts.begin(), &keyPoint2cvKeyPoint<false>);
        return cvKPts;
    };
    const std::vector<cv::KeyPoint> firstKPts = toCvKPts(kptsFirstHolder, first->nbKPoints);
    const std::vector<cv::KeyPoint> secondKPts = toCvKPts(kptsSecondHolder, second->nbKPoints);
    std::vector<cv::DMatch> cvMatches(matches.kptsMatches.size());
    std::transform(matches.kptsMatches.begin(), matches.kptsMatches.end(), cvMatches.begin(), [](const std::pair<Index, Index>& m){
        return cv::DMatch{int(m.first), int(m.second), 1.f};
    });
    cv::Mat imgFirst, imgSecond;
    cv::resize(cv::imread(first->file), imgFirst, cv::Size{first->width, first->height});
    cv::resize(cv::imread(second->file), imgSecond, cv::Size{second->width, second->height});
    cv::Mat imgMatches;
    cv::drawMatches(imgFirst, firstKPts, imgSecond, secondKPts, cvMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    printf("drawing match %d - %d\n", static_cast<uint32_t>(second->hImage), static_cast<uint32_t>(first->hImage));
    cv::imwrite(fs::path(config().cacheFolder)/makeFmtStr("%s_%03u_%03u.jpg", prefix, static_cast<uint32_t>(second->hImage), static_cast<uint32_t>(first->hImage)), imgMatches);
}

void Builder::notifyProgress() {
    std::lock_guard<std::mutex> lk{mProgressCallbackLock};
    if (mProgressCallback != nullptr) {
        mProgressCallback(mProgressCallbackData);
    }
}

}
