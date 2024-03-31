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

#include "DebugModelViewer.h"
#include <boost/dll.hpp>
#include <boost/version.hpp>
#include <cstdlib>

namespace dll = boost::dll;

DebugModelViewer::DebugModelViewer()
{
}

void DebugModelViewer::initialize() {
#if BOOST_VERSION <= 107500
#define IMPORT_FUNC import
#else
#define IMPORT_FUNC import_symbol
#endif
    mCreator = dll::IMPORT_FUNC<IModelViewerPlugin*(int argc, char* argv[])>(
        "/home/yao/projects/rsfm_online/rsfm_client/build/Release/libModelViewer.so",
        "createModelViewerPlugin", dll::load_mode::rtld_lazy);
    static int argc = 1;
    static char* arg0 = const_cast<char*>("DebugModelViewer");
    mViewer.reset(mCreator(argc, &arg0));
}

void DebugModelViewer::setModel(const rsm::RapidSparseModel& model, bool autoCenter, uint32_t idxHighlightCap)
{
    if (mViewer == nullptr) {
        initialize();
    }
    mViewer->setModel(model, autoCenter);
	mViewer->highlightCapture(idxHighlightCap);
}
