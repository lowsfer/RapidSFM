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

#include <iostream>
#include "../Builder.h"
#include <FiberUtils.h>
#include <cuda_utils.h>
#include "../Config.h"
#include <signal.h>
#include <cstdlib>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pwd.h>
#include <Profiler.h>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include "../exifUtils.h"
#include <GeographicLib/LocalCartesian.hpp>
#include <WatchDog.h>
namespace po = boost::program_options;
using namespace std;
using namespace rsfm;

void disableCoreDump() {
  struct rlimit rlim {0, 0};
  setrlimit(RLIMIT_CORE, &rlim);
}

void syncCuda(int sig) {
    disableCoreDump();
    signal(sig, SIG_IGN);
    unused(cudaDeviceSynchronize());
    exit(EXIT_FAILURE);
}

struct InputConfig {
    std::vector<std::string> profiles;
    struct Intrinsics {
        uint32_t width;
        uint32_t height;
        float fx, fy, cx, cy, k1, k2, p1, p2, k3;
    };
    std::vector<Intrinsics> intrinsics;
    struct Image {
        uint32_t idxIntri;
        std::string imgFile;
        std::optional<double3> gnss;
    };
    std::vector<Image> images;
    std::optional<GPSLoc> gnssBase;
};

InputConfig parseArgs(int argc, const char* argv[]) {
    InputConfig inputCfg;
    auto const exe = std::string{argv[0]};
    po::options_description desc("Example:\n  " + exe + " -f 2311 -d ${folder_for_images}\n  " + exe + " --useExifFocal -d ${folder_for_images}\nAllowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("profile,p", po::value<std::vector<std::string>>(), "profile config for builder")
        ("optimizeIntri,o", po::value<bool>(), "optimize intrinsics or not (fixed)")
        ("crossCheck,c", po::value<bool>(), "cross-check feature matching")
        ("nbKPts,k", po::value<uint32_t>(), "target number of key points per image")
        ("intriType,i", po::value<std::underlying_type_t<IntriType>>(),
            "0 - kF1, i.e. {f};\n"
            "1 - kF2, i.e. {fx, fy};\n"
            "2 - kF2C2, i.e. f, cx, cy;\n"
            "3 - kF1D2, i.e. f, k1, k2;\n"
            "4 - kF1D5, i.e. f, k1, k2, p1, p2, k3;\n"
            "5 - kF1C2D5, i.e. f, cx, cy, k1, k2, p1, p2, k3;\n"
            "6 - kF2D5, i.e. fx, fy, k1, k2, p1, p2, k3;\n"
            "7 - kF2C2D5, i.e. fx, fy, cx, cy, k1, k2, p1, p2, k3;")
        ("vocabulary,v", po::value<std::string>(), "Path to the vocabulary path for bad of words. It's recommended to include sub-string \"${sift.descType}\" in the file name for different descriptor types, as the vocabulary is specific for each descriptor type.")
        ("rebuildVoc,r", po::value<bool>(), "force rebuild vocabulary")
        ("saveNewVoc,s", po::value<bool>(), "in case a new vocabulary is built, save it or not")
        ("stackSize", po::value<uint32_t>(), "stack size for fibers")
        ("useExifFocal,e", "use 35mm equivalent focal length from image exif, lower priority than --f, --fx and --fy")
        ("useExifGPS,g", "use GPS (WGS-84) from exif")
        ("perImgIntri", "whether intrinsics are per image or shared among images in the same folder")
        ("f,f", po::value<float>(), "focal length, lower priority than --fx and --fy")
        ("fx", po::value<float>(), "horizontal focal length")
        ("fy", po::value<float>(), "vertical focal length")
        ("cx", po::value<float>(), "horizontal optical center")
        ("cy", po::value<float>(), "vertical optical center")
        ("k1", po::value<float>(), "distortion parameter k1")
        ("k2", po::value<float>(), "distortion parameter k2")
        ("p1", po::value<float>(), "distortion parameter p1")
        ("p2", po::value<float>(), "distortion parameter p2")
        ("k3", po::value<float>(), "distortion parameter k3")
        ("directory,d", po::value<std::vector<std::string>>(), "folder containing input images (no recursion)")
        ("useScheduler", po::value<bool>(), "reorder images before adding to 3D model with a scheduler")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (argc == 1 || vm.count("help")) {
        cout <<desc << "\n";
        exit(EXIT_SUCCESS);
    }

    if (vm.count("profile")) {
        for (const auto& f : vm["profile"].template as<std::vector<std::string>>()) {
            inputCfg.profiles.emplace_back(loadTextFile(f));
        }
    }

    {
        using YAML::Node;
        Node root;
        Node cfg = root["config"];
        if (vm.count("optimizeIntri")) {
            cfg["bundle"]["optimizeIntrinsics"] = vm["optimizeIntri"].template as<bool>();
        }
        if (vm.count("crossCheck")) {
            cfg["pair"]["crossCheck"] = vm["crossCheck"].template as<bool>();
        }
        if (vm.count("nbKPts")) {
            cfg["sift"]["targetNbKPoints"] = vm["nbKPts"].template as<uint32_t>();
        }
        if (vm.count("intriType")) {
            static std::vector<std::string> const num2IntriType{"kF1", "kF2", "kF2C2", "kF1D2", "kF1D5", "kF1C2D5", "kF2D5", "kF2C2D5"};
            cfg["opticModel"] = num2IntriType.at(vm["intriType"].template as<uint32_t>());
        }
        if (vm.count("vocabulary")) {
            cfg["vocabulary"]["path"] = vm["vocabulary"].template as<std::string>();
        }
        if (vm.count("rebuildVoc")) {
            cfg["vocabulary"]["rebuild"] = vm["rebuildVoc"].template as<bool>();
        }
        if (vm.count("saveNewVoc")) {
            cfg["vocabulary"]["saveAfterRebuild"] = vm["saveNewVoc"].template as<bool>();
        }
        if (vm.count("stackSize")) {
            cfg["fiberPool"]["stackSize"] = vm["stackSize"].template as<uint32_t>();
        }
        if (vm.count("useScheduler")) {
            cfg["model"]["useScheduler"] = vm["useScheduler"].template as<bool>();
        }
        std::ostringstream ss;
        ss << root;
        inputCfg.profiles.emplace_back(ss.str());
    }

    const bool useExifFocal = bool(vm.count("useExifFocal")) && !vm.count("f") && !(vm.count("fx") && vm.count("fy"));
    const bool useExifGPS = bool(vm.count("useExifGPS"));

    if (!vm.count("useExifFocal") && !vm.count("f") && !(vm.count("fx") && vm.count("fy"))) {
        fprintf(stderr, "Error: Focal length is not provided. You can use --useExifFocal to detect from EXIF tags, or explicitly specify it with -f, or -fx/-fy.\n");
        throw std::runtime_error("bad arguments");
    };

    const bool perImgIntri = bool(vm.count("perImgIntri"));
    InputConfig::Intrinsics intri {kInvalid<uint32_t>, kInvalid<uint32_t>, NAN, NAN, NAN, NAN, 0, 0, 0, 0, 0};
    if (vm.count("f")) {
        intri.fx = intri.fy = vm["f"].template as<float>();
    }
#define PARSE_INTRI_PARAM(x) do{if (vm.count(#x)) {intri.x = vm[#x].template as<float>();}}while(false)
    PARSE_INTRI_PARAM(fx);
    PARSE_INTRI_PARAM(fy);
    PARSE_INTRI_PARAM(cx);
    PARSE_INTRI_PARAM(cy);
    PARSE_INTRI_PARAM(k1);
    PARSE_INTRI_PARAM(k2);
    PARSE_INTRI_PARAM(p1);
    PARSE_INTRI_PARAM(p2);
    PARSE_INTRI_PARAM(k3);
#undef PARSE_INTRI_PARAM
    ASSERT(std::isnan(intri.fx) == std::isnan(intri.fy));

    std::optional<GeographicLib::LocalCartesian> geoConverter;
    if (!vm.count("directory")) {
        DIE("No input images");
    }
    for (const auto& dir : vm["directory"].template as<std::vector<std::string>>()) {
        ASSERT(fs::is_directory(dir));
        if (!perImgIntri) {
            inputCfg.intrinsics.push_back(intri);
        }
        bool isFirstInDir = true;
        std::vector<fs::path> imgFiles;
        for(const auto& file : fs::directory_iterator(dir)) {
            const auto& filepath = file.path();
            if (!fs::is_regular_file(filepath)) {
                continue;
            }
            std::string ext = filepath.extension();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](char x){return std::tolower<char>(x, std::locale{});});
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".avif" && ext != ".heif") {
                continue;
            }
            imgFiles.emplace_back(filepath);
        }
        std::sort(imgFiles.begin(), imgFiles.end());
        for(const auto& filepath : imgFiles) {
            if (perImgIntri) {
                inputCfg.intrinsics.push_back(intri);
            }
            auto& imgIntri = inputCfg.intrinsics.back();
            std::optional<ExifInfo> exif;
            if (imgIntri.width == kInvalid<uint32_t> || imgIntri.height == kInvalid<uint32_t> || useExifGPS || (useExifFocal && (isFirstInDir || perImgIntri))) {
                exif = getExifInfo(filepath);
            }
            if (imgIntri.width == kInvalid<uint32_t>) {
                imgIntri.width = exif->width;
            }
            if (imgIntri.height == kInvalid<uint32_t>) {
                imgIntri.height = exif->height;
            }
            if (useExifFocal && (std::isnan(imgIntri.fx) || std::isnan(imgIntri.fy))) {
                if (!exif->f.has_value() || !std::isfinite(exif->f.value())) {
                    throw std::runtime_error("Failed to detect focal length from EXIF. Please provide it with -f or -fx/-fy instead.");
                }
                imgIntri.fx = imgIntri.fy = exif->f.value();
            }
            if (std::isnan(imgIntri.cx)) {
                imgIntri.cx = imgIntri.width * 0.5f;
            }
            if (std::isnan(imgIntri.cy)) {
                imgIntri.cy = imgIntri.height * 0.5f;
            }
            inputCfg.images.emplace_back(InputConfig::Image{cast32u(inputCfg.intrinsics.size() - 1), filepath.string(), std::nullopt});
            if (useExifGPS) {
                ASSERT(exif->gnss.has_value() && "GPS not detected in EXIF");
                if (!inputCfg.gnssBase) {
                    inputCfg.gnssBase = exif->gnss.value();
                    geoConverter = GeographicLib::LocalCartesian{inputCfg.gnssBase->latitude, inputCfg.gnssBase->longitude, inputCfg.gnssBase->altitude, GeographicLib::Geocentric::WGS84()};
                }
                double3 cart;
                geoConverter->Forward(exif->gnss->latitude, exif->gnss->longitude, exif->gnss->altitude, cart.x, cart.y, cart.z);
                inputCfg.images.back().gnss = cart;
            }
            isFirstInDir = false;
        }
    }
    return inputCfg;
}

int main(int argc, const char* argv[])
{
    const InputConfig inCfg = parseArgs(argc, argv);

    for (const auto sig : {SIGINT}) {
        signal(sig, syncCuda);
    }
    
    cudaCheck(cudaFree(nullptr));
    size_t nbTasks = 0;

    std::vector<const char*> yamlDocs(inCfg.profiles.size());
    std::transform(inCfg.profiles.begin(), inCfg.profiles.end(), yamlDocs.begin(), [](const std::string& x){return x.c_str();});
    std::unique_ptr<IBuilder> builder {createBuilder(yamlDocs.data(), yamlDocs.size())};

    cudapp::WatchDog watchDog{90.f};
    builder->setProgressCallback([](void* watchdog){
        static_cast<cudapp::WatchDog*>(watchdog)->notifyAlive();
    }, &watchDog);

    std::vector<CameraHandle> hCameras;
    for (const auto& intri : inCfg.intrinsics) {
        const auto hCam = builder->addCamera(intri.width, intri.height, intri.fx, intri.fy, intri.cx, intri.cy, intri.k1, intri.k2, intri.p1, intri.p2, intri.k3);
        hCameras.push_back(hCam);
    }

    for (const auto& img : inCfg.images) {
        const auto hPose = img.gnss ? builder->addPose(img.gnss->x, img.gnss->y, img.gnss->z) : builder->addPose();
        const auto hImage = builder->addImage(img.imgFile.c_str(), hCameras.at(img.idxIntri), hPose, nullptr, 0);
        unused(hImage);
        nbTasks++;
    }

    std::atomic_bool isFinished{false};
    auto progressReportThrd = std::async(std::launch::async, [&](){
        std::vector<size_t> pipelineStatus;
        while (true){
            if (isFinished.load()) {
                break;
            }
            pipelineStatus.resize(builder->getNbPipelineStages() + 1);
            builder->getPipelineStatus(pipelineStatus.data());
            std::stringstream ss;
            ss << "Pipeline Status:";
            for (const auto s : pipelineStatus) {
                ss << '\t' << s;
            }
            // ss << makeFmtStr(", %lu pending fibers, %lu blocking tasks", builder->getNbPendingFibers(), builder->getNbBlockingTasks());
            if (isFinished.load()) {
                break;
            }
            std::cout << ss.str() << std::endl;
            if (pipelineStatus.back() == nbTasks){
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
    builder->finish();
    isFinished.store(true);
    builder->writeClouds("cloud_", IBuilder::kPLY | IBuilder::kNVM | IBuilder::kRSM);

    builder.reset();

    cudapp::Profiler::instance().printSummary(std::cout);
    return 0;
}
