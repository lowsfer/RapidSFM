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
#include "../IModelViewerPlugin.h"
#include <functional>
#include <memory>

class DebugModelViewer
{
public:
    DebugModelViewer();
    void setModel(const rsm::RapidSparseModel& model, bool autoCenter, uint32_t idxHighlightCap);
private:
    void initialize();
private:
    std::function<IModelViewerPlugin*(int argc, char* argv[])> mCreator; // also holds strong ref to the dll handle
    std::unique_ptr<IModelViewerPlugin> mViewer;
};
