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
#include <cstddef>
#include <cstdint>
#include <string>
#include <cmath>
#include "RapidSFM.h"
#include <thread>
#include "RapidBA.h"
#include <public_types.h>
#include <stdexcept>
#include <sys/types.h>
#include <pwd.h>
namespace rsfm
{

using ShutterType = rba::ShutterType;
struct Config
{
    std::string cacheFolder = "./rsfm_cache";

    IntriType opticModel = IntriType::kF1D5;

	rba::ShutterType shutterType = rba::ShutterType::kGlobal;
	bool rollingShutterOnlyInFinalBA = true;

#if 0
    struct MemPool{
        size_t deviceBytes = 64lu << 20;
        size_t pinnedBytes = 128lu << 20;
        size_t sysBytes = 256lu << 20;
    } memPool;

    struct ObjCache{
        size_t deviceBytes = 64lu << 20;
        size_t pinnedBytes = 128lu << 20;
        size_t sysBytes = 256lu << 20;
        // Disk cache space is assumed to be infinite
    } objCache;
#else
    struct MemPool{
        size_t deviceBytes = 6lu << 30;
        size_t pinnedBytes = 12lu << 30;
        size_t sysBytes = 40lu << 30;
    } memPool;

    struct ObjCache{
        size_t deviceBytes = 4lu << 30;
        size_t pinnedBytes = 8lu << 30;
        size_t sysBytes = 32lu << 30;
        // Disk cache space is assumed to be infinite
    } objCache;
#endif
    struct Vocabulary{
        std::string path = "~/rapidsfm/${sift.descType}.vocabulary";
        bool rebuild = false; // Rebuild if rebuild==true, or path is an invalid vocabulary.
        bool saveAfterRebuild = true;
        uint32_t maxRebuildImages = 1000;
        uint32_t maxRebuildKPoints = 1u<<23; // 1GB memory for 128-byte descriptors
    } vocabulary;

    std::string getVocabularyPath() const {
        static std::string_view const descTypePattern = "${sift.descType}";
        static char const* const descType = [this]() -> char const* {
            switch (sift.descType) {
            case DescType::kSIFT: return "sift";
            case DescType::kRootSIFT: return "rootsift";
            case DescType::kSOSNet: return "sosnet";
            }
            throw std::runtime_error("sift.descType is invalid");
        }();
        static std::string_view homedir = []{
            const char *homedir;
            if ((homedir = getenv("HOME")) == NULL) {
                homedir = getpwuid(getuid())->pw_dir;
            }
            return homedir;
        }();
        static std::pair<std::string_view, std::string_view> const replaceMap[] = {
            {descTypePattern, descType},
            {"~", homedir}
        };
        std::string ret = vocabulary.path;
        for (auto& [pattern, val] : replaceMap) {
            size_t pos = 0;
            while (true) {
                pos = ret.find(pattern, pos);
                if (pos == std::string::npos) {
                    break;
                }
                ret.replace(pos, pattern.size(), val);
            }
        }
        return ret;
    }

    uint32_t nbRandStreams = 16;

    struct Sift{
        uint32_t nbWorkers = 2;
        uint32_t targetNbKPoints = 5000u;
        float initContrastThreshold = 0.04f;
        float minContrastThreshold = 0.002f;
        float minOverDetectRatio = 4.f;
        // Prefer false if it can provide enough key points. Up-sample reduces repeatibility.
        bool upSample = false;
        bool uniform = true; // only true is supported for now
        bool detectForOblique = false;
        DescType descType = DescType::kRootSIFT;
    } sift;

    struct BoW{
        uint32_t nbAbstractSamples = 5000;
    } bow;

    struct Matcher{
        uint32_t nbWorkers = 1;
        bool useSift4b = false;
		enum Method{
			kBruteForce,
			kCascadeHash
		};
		Method method = kBruteForce;
    } matcher;

    struct MatchFilter{
        bool useGPU = true;
        // Only the GPU filter respects the settings below
        uint32_t nbRansacTests = 256;
        uint32_t cellCols = 16;
        uint32_t nbWorkers = 4;
        float relativeThreshold = 0.15f; // relative to cell width
        uint32_t minVotes = 2;
        bool tryOtherAffines = true; // Whether we should try affine transformation solutions of all other cells
    } matchFilter;

    struct Pair
    {
        uint32_t maxNbCandidates = 40;
        uint32_t nbNeighbours = 4;
        // below is for secondary candidates. Currently not used
        // expectedNbPairing = expectedNbPairing * attenuation + newNbPairing * (1-attenuation)
        float attenuation = 0.8f;
        float overPairingFactor = 1.2f; // try to solve std::max(expectedNbPairing+minNbOverPairing, expectedNbPairing * overPairing) pairs
        uint32_t minNbOverPairing = 2;

        bool crossCheck = true;
        uint32_t pmfMinVotes = 2;
    } pair;

    struct PairSolver
    {
        bool useDegensac = true;
        float requiredRansacConfidence = 0.995f;
        bool preferAccuracy = true; // false is perfer performance.
        float epipolarityRansacRelativeThreshold = 0.008f;
        float homographyRansacRelativeThreshold = 0.005f;
		float tiePtWeight = 1.f;
        float disambiguityFrontRatio = 0.95f; // all H or F solutions must have at least such ratio of inliners in front of both cameras. Relative to the solution with most front inliers.
        float disambiguityNbInliersRatio = 0.625f; // for H or F decomposition
        float duplicateRotationThreshold = 5.f / 180.f * M_PI;
        uint32_t minNbInliers = 50;
        float minSpan = 0.292f * 0.15f;
        float refineRelativeHuberDelta = 0.005f; // error in two dimensions
        float recheckRelativeThreshold = 0.01f; // error in two dimensions
        uint32_t nbRefineIterations = 10; // 2 may be insufficient for convergence.
        float refineConvergThresAngle = 2.f / 180.f * M_PI;
        float zMin = 0.02f; // relative to translation length
        float zMax = 128.f;
        float minAngle = 0.1f / 180 * M_PI; // in rad
        float maxAngle = 90.f / 180 * M_PI; // in rad
        // fundamental matrix RT has low accurancy when overlap is < 40% and it cannot be recovered by RT optimization. No idea why.
        bool skipHomographyIfFundamentalMatrixIsGood = false;
        float solutionDisambiguityScoreRatio = 0.8f; // for final solutions
    } pairSolver;

    struct PnPSolver
    {
        float requiredRansacConfidence = 0.999f;
        float ransacRelativeThreshold = 0.01f;
        float ransacSampleMinSpan = 0.292f * 0.15f;
        uint32_t maxNbRansacTests = 5000U;
        float minSpan = 0.292f * 0.15f;
		float tiePtWeight = 1.f;
        uint32_t minNbInliers = 50;
        float minInlierRatio = 0.33f;
        bool optimizeWithBA = true;
    } pnpSolver;

    struct FiberBlockingService{
        uint32_t interval = 100; // in ms
        uint32_t windowSize = 1024;
        uint32_t maxPendingTasks = 1u<<20;
    } blockingService;

    struct FiberPool{
        uint32_t nbThreads = std::thread::hardware_concurrency();
        size_t stackSize = 1024*256; // in Bytes. Seems 16KB is not enough and 32KB is OK. 4KB guard page will be padded.
    } fiberPool;

    struct Model{
        bool useScheduler = false; // false means add images in input order. Otherwise collect all image pairs and re-order images with the scheduler.
        uint32_t minPairInliersForInit = 300u;
        float maxMedianDepthForInit = 32;
        float minMedianAngleForInit = 0.5f/180*M_PI; // 2.f/180*M_PI
        uint32_t minNbCommonPoints = 4u; // For image adding. Duplicate. Already covered by pnpSolver.minNbInliers
        float triangulationReprojHuberRelative = 0.005f;
        // Skip location updating in IncreModel::updatePointWithLastOb() when a point has more than nbStableObs observations.
        uint32_t nbStableObs = 8u;
        float zMin = 0.1f; // relative to the translation length of the first pair in this model
        float zMax = 1000.f;
        float cosMinAngle = std::cos(0.5f/180*M_PI);
        uint32_t intervalLocalBA = 8;
        float ratioGlobalBA = 1.25; // perform global BA when number of fused images are increased by this ratio.
        bool autoScale = false; // make average depth around 5.0f. No longer needed as we are using proper relative thresholding for both rsfm and rmvs.
        float goodPairConstructionThreshold = 0.8f; // ratio of points constructed in the model for inliers of a pair. for re-triangulation. Colmap uses 0.2 but the base is all matches.
        struct Merge
        {
            uint32_t minNbCommonPoints = 500u; // for model merge
            uint32_t nbLinksForForcedMerge = 16u; // If we have many link pairs, we ignore the minNbCommonPoints requirement.
            uint32_t forcedMergeMinNbCommonPoints = 20u;
            float ransacConfidence = 0.999f;
            float ransacThreshold = 0.05f; // relative to point depth to camera
            uint32_t maxNbRansacTests = 5000U;
            float minInlierRatio = 0.5f;
        } merge;
    } model;

    struct Bundle{
        bool useGrpModel = true;
        float huberLocal = 3.f;
        float huberGlobal = 3.f;
		float tiePtWeight = 1.f;
        bool applyForFirstThreeImages = false; // apply BA for the first 3 images.
        uint32_t useGlobalIfNoLargerThan = 0;
        bool optimizeIntrinsics = true;
        uint32_t minNbCapturesForIntrinsicsOptimization = 128; //  Final global BA ignores this
        float ctrlPtObOmega = 16.f;
    } bundle;
};

}
