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
#include <atomic>
#include <type_traits>
#include "fwd.h"
#include <macros.h>

namespace rsfm{
/*
// Declared in fwd.h
template <typename Handle, typename = void>
class HandleGenerator;
*/
template <typename Handle>
class HandleGenerator<Handle, std::enable_if_t<std::is_integral_v<Handle>, void>>
{
public:
    Handle make() {
        const Handle h = mIdxNext.fetch_add(1, std::memory_order_relaxed);
        ASSERT(h != kInvalid<Handle>);
        return h;
    }
    Handle peekAtNext() const {return mIdxNext.load(std::memory_order_relaxed);}
    void skip(Handle x) {mIdxNext.fetch_add(x, std::memory_order_relaxed); }
private:
    std::atomic<Handle> mIdxNext {0};
};

template <typename Handle>
class HandleGenerator<Handle, std::enable_if_t<std::is_enum_v<Handle>, void>>
{
public:
    Handle make() { return static_cast<Handle>(mGenerator.make()); }
    Handle peekAtNext() const {return static_cast<Handle>(mGenerator.peekAtNext()); }
    void skip(std::underlying_type_t<Handle> x) {mGenerator.skip(x); }
private:
    HandleGenerator<std::underlying_type_t<Handle>> mGenerator;
};

} // namespace rsfm
