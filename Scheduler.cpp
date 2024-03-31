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

#include "Scheduler.h"

#include "ImagePair.h"
#include <unordered_set>
#include <macros.h>

namespace rsfm
{

void Scheduler::add(ImageHandle hImage, const std::vector<std::unique_ptr<ImagePair>>& pairs) {
    if (pairs.empty()) {
        mGraph[hImage];
    }
    for (const auto& p : pairs) {
        const auto [hFirst, hSecond] = p->images;
        const float score = p->solutions.front().score;
        mGraph[hFirst][hSecond] = score;
        mGraph[hSecond][hFirst] = score;
    }
}

std::vector<ImageHandle> Scheduler::findInitImage() const {
    std::vector<std::pair<ImageHandle, float>> initScores;
    initScores.reserve(mGraph.size());
    std::unordered_set<ImageHandle, Hasher> visited;
    std::vector<ImageHandle> wavefront;
    std::vector<ImageHandle> wavefrontNext;
    for (const auto& [hImage, edges] : mGraph) {
        visited.clear();
        wavefront.clear();
        wavefront.push_back(hImage);
        visited.emplace(hImage);
        wavefrontNext.clear();
        float score = 0.f;
        for (int32_t i = 0; i < 4 && !wavefront.empty(); i++) {
            for (const auto& u : wavefront) {
                for (const auto& [v, s] : mGraph.at(u)) {
                    if (visited.find(v) != visited.end()) {
                        continue;
                    }
                    wavefrontNext.emplace_back(v);
                    visited.emplace(v);
                    score += s * std::pow(0.5f, i);
                }
            }
            wavefront.swap(wavefrontNext);
            wavefrontNext.clear();
        }  
        initScores.emplace_back(hImage, score);
    }
    std::sort(initScores.begin(), initScores.end(), [](auto& x, auto& y){return x.second > y.second;});
    std::vector<ImageHandle> result(initScores.size());
    std::transform(initScores.begin(), initScores.end(), result.begin(), [](auto& x){return x.first;});
    ASSERT(result.size() == mGraph.size());
    return result;
}

std::vector<ImageHandle> Scheduler::getSequence() const {
    std::vector<ImageHandle> seq;
    std::unordered_set<ImageHandle, Hasher> visited;
    std::unordered_set<ImageHandle> wavefront;
    std::unordered_map<ImageHandle, float> candidates;
    std::vector<ImageHandle> toBeRemovedFromWavefront;

    const std::vector<ImageHandle> initImagePreference = findInitImage();
    while (seq.size() != mGraph.size()) {
        wavefront.clear();
        candidates.clear();
        const auto iter = std::find_if(initImagePreference.begin(), initImagePreference.end(), [&](ImageHandle h){return visited.find(h) == visited.end();});
        ASSERT(iter != initImagePreference.end());
        const ImageHandle hInit = *iter;
        seq.emplace_back(hInit);
        wavefront.emplace(hInit);
        visited.emplace(hInit);
        while (!wavefront.empty()) {
            candidates.clear();
            toBeRemovedFromWavefront.clear();
            for (const auto& u : wavefront) {
                bool isValidWavefront = false;
                for (const auto& [v, s] : mGraph.at(u)) {
                    if (visited.find(v) != visited.end()) {
                        continue;
                    }
                    candidates[v] += s;
                    isValidWavefront = true;
                }
                if (!isValidWavefront) {
                    toBeRemovedFromWavefront.emplace_back(u);
                }
            }
            for (const auto& h : toBeRemovedFromWavefront) {
                wavefront.erase(h);
            }
            if (candidates.empty()) {
                ASSERT(wavefront.empty());
                break;
            }
            const auto iterMax = std::max_element(candidates.begin(), candidates.end(), [](auto& x, auto& y){return x.second < y.second;});
            ASSERT(iterMax != candidates.end());
            const ImageHandle hNext = iterMax->first;
            seq.emplace_back(hNext);
            wavefront.emplace(hNext);
            visited.emplace(hNext);
        }
    }
    ASSERT(seq.size() == mGraph.size());
    return seq;
}

}