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

#include <yaml-cpp/yaml.h>
#include "../Config.h"
#include <unordered_map>
#include <macros.h>
#include <boost/preprocessor.hpp>
#include "parser.h"

namespace rsfm
{
// Every parse(T& dst, const YAML::Node& node) function updates existing dst
// with content in node.

YAML::Node at(const YAML::Node& node, const char* key) {
    const auto result = node[key];
    ASSERT(result.IsDefined() && "required key is not found");
    return result;
}

#define PARSE_NODE(x) parse(dst.x, node[#x])
#define PARSE_REQUIRED_NODE(x) parse(dst.x, at(node, #x))
#define PARSE_NODE_CALLBACK(r, data, component) PARSE_NODE(component);
#define PARSE_REQUIRED_NODE_CALLBACK(r, data, component) PARSE_REQUIRED_NODE(component);

template <typename T>
void parse(T& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    dst = node.as<T>();
}
template <typename T>
void parse(std::vector<T>& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    ASSERT(node.IsSequence() && dst.size() == node.size());
    auto dstIter = dst.begin();
    for (auto iter = node.begin(); iter != node.end(); iter++) {
        parse(*dstIter++, *iter);
    }
}
template <typename T, size_t length>
void parse(T(&dst)[length], const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    ASSERT(node.IsSequence() && length == node.size());
    auto dstIter = std::begin(dst);
    for (auto iter = node.begin(); iter != node.end(); iter++) {
        parse(*dstIter++, *iter);
    }
}
void parse(IntriType& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    static const std::unordered_map<std::string, IntriType> map {
        {"kF1", IntriType::kF1},
        {"kF2", IntriType::kF2},
        {"kF2C2", IntriType::kF2C2},
        {"kF1D2", IntriType::kF1D2},
        {"kF1D5", IntriType::kF1D5},
        {"kF2D5", IntriType::kF2D5},
        {"kF1C2D5", IntriType::kF1C2D5},
        {"kF2C2D5", IntriType::kF2C2D5}
    };
    dst = map.at(node.as<std::string>());
}

void parse(rba::ShutterType& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    static const std::unordered_map<std::string, rba::ShutterType> map {
        {"kGlobal", rba::ShutterType::kGlobal},
        {"kRollingFixedVelocity", rba::ShutterType::kRollingFixedVelocity},
        {"kRolling1D", rba::ShutterType::kRolling1D},
		{"kRolling1DLoc", rba::ShutterType::kRolling1DLoc},
        {"kRolling3D", rba::ShutterType::kRolling3D}
    };
    dst = map.at(node.as<std::string>());
}

void parse(DescType& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    static const std::unordered_map<std::string, DescType> map {
        {"kSIFT", DescType::kSIFT},
        {"kRootSIFT", DescType::kRootSIFT},
        {"kSOSNet", DescType::kSOSNet}
    };
    dst = map.at(node.as<std::string>());
}
void parse(Config::MemPool& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(deviceBytes);
    PARSE_NODE(pinnedBytes);
    PARSE_NODE(sysBytes);
}
void parse(Config::ObjCache& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(deviceBytes);
    PARSE_NODE(pinnedBytes);
    PARSE_NODE(sysBytes);
}
void parse(Config::Vocabulary& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(path);
    PARSE_NODE(rebuild);
    PARSE_NODE(saveAfterRebuild);
    PARSE_NODE(maxRebuildImages);
    PARSE_NODE(maxRebuildKPoints);
}
void parse(Config::Sift& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(nbWorkers);
    PARSE_NODE(targetNbKPoints);
    PARSE_NODE(initContrastThreshold);
    PARSE_NODE(minContrastThreshold);
    PARSE_NODE(minOverDetectRatio);
    PARSE_NODE(upSample);
    PARSE_NODE(uniform);
    PARSE_NODE(descType);
}
void parse(Config::BoW& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(nbAbstractSamples);
}
void parse(Config::Matcher& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(nbWorkers);
    PARSE_NODE(useSift4b);
}
void parse(Config::Pair& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(maxNbCandidates);
    PARSE_NODE(nbNeighbours);
    PARSE_NODE(attenuation);
    PARSE_NODE(overPairingFactor);
    PARSE_NODE(minNbOverPairing);
    PARSE_NODE(crossCheck);
    PARSE_NODE(pmfMinVotes);
}
void parse(Config::PairSolver& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(useDegensac);
    PARSE_NODE(requiredRansacConfidence);
    PARSE_NODE(preferAccuracy);
    PARSE_NODE(epipolarityRansacRelativeThreshold);
    PARSE_NODE(homographyRansacRelativeThreshold);
	PARSE_NODE(tiePtWeight);
    PARSE_NODE(disambiguityFrontRatio);
    PARSE_NODE(disambiguityNbInliersRatio);
    PARSE_NODE(duplicateRotationThreshold);
    PARSE_NODE(minNbInliers);
    PARSE_NODE(minSpan);
    PARSE_NODE(refineRelativeHuberDelta);
    PARSE_NODE(recheckRelativeThreshold);
    PARSE_NODE(nbRefineIterations);
    PARSE_NODE(refineConvergThresAngle);
    PARSE_NODE(zMin);
    PARSE_NODE(zMax);
    PARSE_NODE(minAngle);
    PARSE_NODE(maxAngle);
    PARSE_NODE(skipHomographyIfFundamentalMatrixIsGood);
    PARSE_NODE(solutionDisambiguityScoreRatio);
}
void parse(Config::PnPSolver& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(requiredRansacConfidence);
    PARSE_NODE(ransacRelativeThreshold);
    PARSE_NODE(ransacSampleMinSpan);
    PARSE_NODE(maxNbRansacTests);
    PARSE_NODE(minSpan);
	PARSE_NODE(tiePtWeight);
    PARSE_NODE(minNbInliers);
    PARSE_NODE(minInlierRatio);
    PARSE_NODE(optimizeWithBA);
}
void parse(Config::FiberBlockingService& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(interval);
    PARSE_NODE(windowSize);
    PARSE_NODE(maxPendingTasks);
}
void parse(Config::FiberPool& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(nbThreads);
    PARSE_NODE(stackSize);
}
void parse(Config::Model::Merge& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(minNbCommonPoints);
    PARSE_NODE(nbLinksForForcedMerge);
    PARSE_NODE(forcedMergeMinNbCommonPoints);
    PARSE_NODE(ransacConfidence);
    PARSE_NODE(ransacThreshold);
    PARSE_NODE(maxNbRansacTests);
    PARSE_NODE(minInlierRatio);
}
void parse(Config::Model& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(minPairInliersForInit);
    PARSE_NODE(maxMedianDepthForInit);
    PARSE_NODE(minMedianAngleForInit);
    PARSE_NODE(minNbCommonPoints);
    PARSE_NODE(triangulationReprojHuberRelative);
    PARSE_NODE(nbStableObs);
    PARSE_NODE(zMin);
    PARSE_NODE(zMax);
    PARSE_NODE(cosMinAngle);
    PARSE_NODE(intervalLocalBA);
    PARSE_NODE(ratioGlobalBA);
    PARSE_NODE(useScheduler);
    PARSE_NODE(autoScale);
    PARSE_NODE(goodPairConstructionThreshold);
    PARSE_NODE(merge);
}
void parse(Config::Bundle& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(useGrpModel);
    PARSE_NODE(huberLocal);
    PARSE_NODE(huberGlobal);
	PARSE_NODE(tiePtWeight);
    PARSE_NODE(useGlobalIfNoLargerThan);
    PARSE_NODE(optimizeIntrinsics);
    PARSE_NODE(minNbCapturesForIntrinsicsOptimization);
    PARSE_NODE(ctrlPtObOmega);
}
void parse(Config::MatchFilter& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(useGPU);
    PARSE_NODE(nbRansacTests);
    PARSE_NODE(cellCols);
    PARSE_NODE(nbWorkers);
    PARSE_NODE(relativeThreshold);
    PARSE_NODE(minVotes);
    PARSE_NODE(tryOtherAffines);
}
void parse(Config& dst, const YAML::Node& node) {
    if (!node.IsDefined()) {
        return;
    }
    PARSE_NODE(cacheFolder);
    PARSE_NODE(opticModel);
	PARSE_NODE(shutterType);
	PARSE_NODE(rollingShutterOnlyInFinalBA);
    PARSE_NODE(memPool);
    PARSE_NODE(objCache);
    PARSE_NODE(vocabulary);
    PARSE_NODE(nbRandStreams);
    PARSE_NODE(sift);
    PARSE_NODE(bow);
    PARSE_NODE(matcher);
    PARSE_NODE(matchFilter);
    PARSE_NODE(pair);
    PARSE_NODE(pairSolver);
    PARSE_NODE(pnpSolver);
    PARSE_NODE(blockingService);
    PARSE_NODE(fiberPool);
    PARSE_NODE(model);
    PARSE_NODE(bundle);
}

void parse(typename Input::Camera& dst, const YAML::Node& node) {
    BOOST_PP_SEQ_FOR_EACH(PARSE_REQUIRED_NODE_CALLBACK, data, (width)(height)(fx)(fy)(cx)(cy)(k1)(k2)(p1)(p2)(k3))
}
void parse(typename Input::Pose& pose, const YAML::Node& node) {
    parse(reinterpret_cast<float(&)[3]>(pose.c.x), node["location"]);
    parse(reinterpret_cast<float(&)[9]>(pose.covariance), node["covariance"]);
    parse(pose.huber, node["huber"]);
}
void parse(typename Input::Image& dst, const YAML::Node& node) {
    BOOST_PP_SEQ_FOR_EACH(PARSE_REQUIRED_NODE_CALLBACK, data, (file)(idxCam)(idxPose))
}
void parse(Input& dst, const YAML::Node& node) {
    BOOST_PP_SEQ_FOR_EACH(PARSE_REQUIRED_NODE_CALLBACK, data, (cameras)(poses)(images))
}

#undef PARSE_REQUIRED_NODE_CALLBACK

void parse(Config& config, const char* yamlDoc) {
    const auto root = YAML::Load(yamlDoc);
    parse(config, root["config"]);
}

void parse(Config& config, const char* yamlDocs[], size_t nbYamlDocs) {
    for (size_t i = 0; i < nbYamlDocs; i++) {
        parse(config, yamlDocs[i]);
    }
}

void parse(Input& input, const char* yamlDoc) {
    const auto root = YAML::Load(yamlDoc);
    parse(input, root["data"]);
}

void parse(Input& input, const char* yamlDocs[], size_t nbYamlDocs) {
    for (size_t i = 0; i < nbYamlDocs; i++) {
        parse(input, yamlDocs[i]);
    }
}
} // namespace rsfm
