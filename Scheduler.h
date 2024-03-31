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
#include "RapidSFM.h"
#include "fwd.h"
#include <unordered_map>
#include <vector>
#include <cpp_utils.h>

namespace rsfm
{
class Scheduler
{
public:
    void add(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& pairs);
    std::vector<ImageHandle> getSequence() const;
private:
    // returns preference
    std::vector<ImageHandle> findInitImage() const;
private:
    using Hasher = DirectMappingHash<ImageHandle>;
    std::unordered_map<ImageHandle, std::unordered_map<ImageHandle, float, Hasher>, Hasher> mGraph; // undirected
};

} // namespace rsfm
