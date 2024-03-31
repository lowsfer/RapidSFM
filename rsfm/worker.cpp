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
#include <Profiler.h>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include "../parser/parser.h"
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
    rsfm::Input input;
};

InputConfig parseArgs(int argc, const char* argv[]) {
    InputConfig inputCfg;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("profile,p", po::value<std::vector<std::string>>(), "profile config for builder")
        ("input,i", po::value<std::string>(), "input yaml config. Some options cannot be used together with this")
        ("progress-socket", po::value<std::string>(), "Unix domain socket for progress reporting")
        ("optimizeIntri,o", po::value<bool>(), "optimize intrinsics or not (fixed)")
        ("crossCheck,c", po::value<bool>(), "cross-check feature matching")
        ("nbKPts,k", po::value<uint32_t>(), "target number of key points per image")
        ("intriType,i", po::value<std::underlying_type_t<IntriType>>(),
            "0 - kF1, i.e. {f};\n"
            "1 - kF2, i.e. {fx, fy};\n"
            "2 - kF2C2, i.e. f, cx, cy;\n"
            "3 - kF1D2, i.e. f, p1, p2;\n"
            "4 - kF1D5, i.e. f, k1, k2, p1, p2, k3;\n"
            "5 - kF1C2D5, i.e. f, cx, cy, k1, k2, p1, p2, k3;\n"
            "6 - kF2D5, i.e. fx, fy, k1, k2, p1, p2, k3;\n"
            "7 - kF2C2D5, i.e. fx, fy, cx, cy, k1, k2, p1, p2, k3;")
        ("rebuildVoc,r", po::value<bool>(), "force rebuild vocabulary")
        ("saveNewVoc,s", po::value<bool>(), "in case a new vocabulary is built, save it or not")
        ("stackSize", po::value<uint32_t>(), "stack size for fibers");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout <<desc << "\n";
        exit(EXIT_SUCCESS);
    }

    if (vm.count("profile")) {
        for (const auto& f : vm["profile"].as<std::vector<std::string>>()) {
            inputCfg.profiles.emplace_back(loadTextFile(f));
        }
    }

    {
        ASSERT(vm.count("input") != 0);
        std::string inputDoc = loadTextFile(vm["input"].as<std::string>());
        parse(inputCfg.input, inputDoc.c_str());
        inputCfg.profiles.emplace_back(std::move(inputDoc));
    }

    {
        using YAML::Node;
        Node root;
        Node cfg = root["config"];
        if (vm.count("optimizeIntri")) {
            cfg["bundle"]["optimizeIntrinsics"] = vm["optimizeIntri"].as<bool>();
        }
        if (vm.count("crossCheck")) {
            cfg["pair"]["crossCheck"] = vm["crossCheck"].as<bool>();
        }
        if (vm.count("nbKPts")) {
            cfg["sift"]["targetNbKPoints"] = vm["nbKPts"].as<uint32_t>();
        }
        if (vm.count("intriType")) {
            static std::vector<std::string> const num2IntriType{"kF1", "kF2", "kF2C2", "kF1D2", "kF1D5", "kF1C2D5", "kF2D5", "kF2C2D5"};
            cfg["intriType"] = num2IntriType.at(vm["intriType"].as<uint32_t>());
        }
        if (vm.count("rebuildVoc")) {
            cfg["vocabulary"]["rebuild"] = vm["rebuildVoc"].as<bool>();
        }
        if (vm.count("saveNewVoc")) {
            cfg["vocabulary"]["saveAfterRebuild"] = vm["saveNewVoc"].as<bool>();
        }
        if (vm.count("stackSize")) {
            cfg["fiberPool"]["stackSize"] = vm["stackSize"].as<uint32_t>();
        }
        std::ostringstream ss;
        ss << root;
        inputCfg.profiles.emplace_back(ss.str());
    }

    return inputCfg;
}

int main(int argc, const char* argv[])
{
    const InputConfig inCfg = parseArgs(argc, argv);

    for (const auto sig : {SIGINT}) {
        signal(sig, syncCuda);
    }

    // const WatchDog watchDog{90.f};
    
    cudaCheck(cudaFree(nullptr));
    size_t nbTasks = 0;

    std::vector<const char*> yamlDocs(inCfg.profiles.size());
    std::transform(inCfg.profiles.begin(), inCfg.profiles.end(), yamlDocs.begin(), [](const std::string& x){return x.c_str();});
    std::unique_ptr<IBuilder> builder {createBuilder(yamlDocs.data(), yamlDocs.size())};

    std::vector<CameraHandle> hCameras;
    for (const auto& cam : inCfg.input.cameras) {
        const auto hCam = builder->addCamera(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
        hCameras.push_back(hCam);
    }

    std::vector<PoseHandle> hPoses;
    for (const auto& pose : inCfg.input.poses) {
        const auto hPose = builder->addPose(pose.r.x, pose.r.y, pose.r.z, pose.c.x, pose.c.y, pose.c.z,
            Covariance3{
                .xx = pose.covariance[0][0], .yy = pose.covariance[1][1], .zz = pose.covariance[2][2],
                .xy = pose.covariance[0][1], .xz = pose.covariance[0][2], .yz = pose.covariance[1][2]
            },
            pose.huber);
        hPoses.emplace_back(hPose);
    }

    for (const auto& img : inCfg.input.images) {
        const auto hImage = builder->addImage(img.file.c_str(), hCameras.at(img.idxCam), hPoses.at(img.idxPose), nullptr, 0);
        unused(hImage);
        nbTasks++;
    }

    while (true){
        std::vector<size_t> nbPendingTasksPerStage(builder->getNbPipelineStages() + 1);
        builder->getPipelineStatus(nbPendingTasksPerStage.data());
        std::stringstream ss;
        ss << "Pipeline Status:";
        for (const auto s : nbPendingTasksPerStage) {
            ss << '\t' << s;
        }
        std::cout << ss.str() << std::endl;
        if (nbPendingTasksPerStage.back() == nbTasks){
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    builder->finish();
    builder->writeClouds("cloud", IBuilder::kPLY | IBuilder::kNVM);

    builder.reset();

    cudapp::Profiler::instance().printSummary(std::cout);
    return EXIT_SUCCESS;
}
