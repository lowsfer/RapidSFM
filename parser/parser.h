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

#include "../Config.h"
#include <yaml-cpp/yaml.h>

namespace rsfm
{

// Update config keys from yamlDoc. yamlDoc is the YAML content, not file name.
void parse(Config& config, const char* yamlDoc);
// Update config keys from multiple docs. Later ones will override previous ones.
void parse(Config& config, const char* yamlDocs[], size_t nbYamlDocs);

struct Input{
    struct Camera {
        uint32_t width = kInvalid<uint32_t>;
        uint32_t height = kInvalid<uint32_t>;
        float fx = kNaN, fy = kNaN;
        float cx = kNaN, cy = kNaN;
        float k1 = kNaN, k2 = kNaN, p1 = kNaN, p2 = kNaN, k3 = kNaN;
    };
    std::vector<Camera> cameras;

    struct Pose {
        float3 r = {kNaN, kNaN, kNaN};
        float3 c = {kNaN, kNaN, kNaN};
        float covariance[3][3] = {kInf, 0, 0, 0, kInf, 0, 0, 0, kInf}; // of c
        float huber = kInf;
    };
    std::vector<Pose> poses;

    struct Image {
        std::string file;
        uint32_t idxCam = kInvalid<uint32_t>;
        uint32_t idxPose = kInvalid<uint32_t>;
    };
    std::vector<Image> images;
};

void parse(Input& input, const char* yamlDoc);

void parse(Input& input, const char* yamlDocs[], size_t nbYamlDocs);
} // namespace rsfm
