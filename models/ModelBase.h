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
#include "../RapidSFM.h"
#include "../fwd.h"
#include "../Types.h"
#include "../Builder.h"

namespace rsfm
{
class ModelBase : public rsfm::IModel
{
public:
    // virtual bool addImage(ImageHandle hImage, const std::vector<std::unique_ptr<Matches>>& allMatches) = 0;
    virtual bool addImage(ImageHandle hImage, std::vector<std::unique_ptr<ImagePair>>& allPairs) = 0;
    virtual void finish() = 0;
    virtual ~ModelBase() = default;
};
} // namespace rsfm
