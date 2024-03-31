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

#include "bruteForceMatch.h"

namespace rsfm
{

std::vector<ValidMatch> crossCheckMatches(const typename SiftBruteForceMatchTask::BestMatch* matches, uint32_t nbQueryDesc,
                                   const typename SiftBruteForceMatchTask::BestMatch* matchesBwd, uint32_t nbTrainDesc){
    std::vector<ValidMatch> valid_matches;
    valid_matches.reserve(std::min(nbQueryDesc, nbTrainDesc));
    for(uint32_t i = 0; i < nbQueryDesc; i++) {
        const uint32_t trainIdx = matches[i].index;
		if (trainIdx == ~0U) {
			continue;
		}
        assert(trainIdx < nbTrainDesc);
        if (matchesBwd[trainIdx].index == i){
            assert(matches[i].distance == matchesBwd[trainIdx].distance);
            valid_matches.emplace_back(ValidMatch{i, matches[i].index, float(matches[i].distance)});
        }
    }
    valid_matches.shrink_to_fit();
    return valid_matches;
};

std::vector<ValidMatch> removeMatchConflicts(const typename SiftBruteForceMatchTask::BestMatch* matches, uint32_t nbQueryDesc, uint32_t nbTrainDesc)
{
    using BestMatch = typename SiftBruteForceMatchTask::BestMatch;
    constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();
    std::vector<BestMatch> matchesBwd(nbTrainDesc, BestMatch{kInvalid, std::numeric_limits<decltype(std::declval<BestMatch>().distance)>::max()});
    for (uint32_t i = 0; i < nbQueryDesc; i++) {
        const BestMatch& q = matches[i];
		if (q.index == kInvalid) {
			continue;
		}
        BestMatch& t = matchesBwd.at(q.index);
        if (q.distance < t.distance)
            t = BestMatch{i, q.distance};
    }
    std::vector<ValidMatch> validMatches;
    validMatches.reserve(std::min(nbQueryDesc, nbTrainDesc));
    for (uint32_t i = 0; i < nbTrainDesc; i++) {
        const auto& t = matchesBwd[i];
        if (t.index != kInvalid) {
            validMatches.emplace_back(ValidMatch{t.index, i, float(t.distance)});
        }
    }
    validMatches.shrink_to_fit();
    return validMatches;
}

} // namespace rsfm