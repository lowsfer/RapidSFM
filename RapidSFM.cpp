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

#include "RapidSFM.h"
#include "Config.h"
#include "parser/parser.h"
#include "Builder.h"

namespace rsfm
{
IModel::~IModel() = default;
IBuilder::~IBuilder() = default;

IBuilder* createBuilder() {
    auto cfg = std::make_unique<Config>();
    return new Builder(std::move(cfg));
}

IBuilder* createBuilder(const char* yamlDoc) {
    auto cfg = std::make_unique<Config>();
    parse(*cfg, yamlDoc);
    return new Builder(std::move(cfg));
}

IBuilder* createBuilder(const char* yamlDocs[], size_t nbDocs) {
    auto cfg = std::make_unique<Config>();
    parse(*cfg, yamlDocs, nbDocs);
    return new Builder(std::move(cfg));
}
}