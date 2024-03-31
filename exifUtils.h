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

#include <cstdint>
#include <optional>
#include <exiv2/exiv2.hpp>
#include <macros.h>
#include <mutex>
#include <filesystem>
#include <cpp_utils.h>
namespace fs = std::filesystem;

struct GPSLoc{
    double latitude; // in degrees
    double longitude; // in degrees
    double altitude; // in meters
};

struct ExifInfo {
    uint32_t width;
    uint32_t height;
    std::optional<float> f;
    std::optional<GPSLoc> gnss;
};

inline ExifInfo getExifInfo(const fs::path& file) {
    ExifInfo result;

    using namespace Exiv2;
    static std::once_flag initFlag;
    std::call_once(initFlag, [](){
        Exiv2::XmpParser::initialize();
        ::atexit(Exiv2::XmpParser::terminate);
    });

    const Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(file.string());
    ASSERT(image.get() != 0);
    image->readMetadata();
    result.width = image->pixelWidth();
    result.height = image->pixelHeight();
 
    Exiv2::ExifData &exifData = image->exifData();
    if (exifData.empty()) {
        return result;
    }
    {
        static const ExifKey f35key {"Exif.Photo.FocalLengthIn35mmFilm"};
        const auto iter = exifData.findKey(f35key);
        if (iter != exifData.end()) {
            const float f35mm = exifData.findKey(f35key)->toFloat();
            result.f = std::sqrt(float(square(result.width) + square(result.height)) / float(36*36+24*24)) * f35mm;
        }
    }
    {
        auto getGpsDegree = [&](const ExifData::iterator& iterRef, const ExifData::iterator& iterVal,
                const char positiveRef, const char negativeRef) -> std::optional<double>{
            if (iterRef == exifData.end() || iterVal == exifData.end()) {
                return {};
            }
            const auto ref = iterRef->toLong();
            ASSERT((ref == positiveRef || ref == negativeRef) && iterVal->count() == 3);
            const int sign = (ref == positiveRef ? 1 : -1);
            const double degree = iterVal->toFloat(0);
            const double minute = iterVal->toFloat(1);
            const double second = iterVal->toFloat(2);
            return (degree + minute / 60.0 + second / 3600.0) * sign;
        };
        static const ExifKey gnssTagLatRef {"Exif.GPSInfo.GPSLatitudeRef"};
        static const ExifKey gnssTagLat {"Exif.GPSInfo.GPSLatitude"};
        const std::optional<double> latitude = getGpsDegree(exifData.findKey(gnssTagLatRef), exifData.findKey(gnssTagLat), 'N', 'S');
        static const ExifKey gnssTagLongiRef {"Exif.GPSInfo.GPSLongitudeRef"};
        static const ExifKey gnssTagLongi {"Exif.GPSInfo.GPSLongitude"};
        const std::optional<double> longitude = getGpsDegree(exifData.findKey(gnssTagLongiRef), exifData.findKey(gnssTagLongi), 'E', 'W');
        static const ExifKey gnssTagAltiRef {"Exif.GPSInfo.GPSAltitudeRef"};
        static const ExifKey gnssTagAlti {"Exif.GPSInfo.GPSAltitude"};
        const std::optional<double> altitude = [&]() -> std::optional<double>{
            std::optional<double> result;
            const auto iterRef = exifData.findKey(gnssTagAltiRef);
            const auto iterVal = exifData.findKey(gnssTagAlti);
            if (iterRef == exifData.end() || iterVal == exifData.end()) {
                return result;
            }
            const auto ref = iterRef->toLong();
            ASSERT((ref == 0 || ref == 1));
            result = iterVal->toFloat() * (ref == 0 ? 1 : -1);
            return result;
        }();
        if (latitude && longitude && altitude)         {
            result.gnss = GPSLoc{latitude.value(), longitude.value(), altitude.value()};
        }
    }
    return result;
}
