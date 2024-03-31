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

#include <eigen3/Eigen/Core>
#include "Types.hpp"
#include <macros.h>
#include "ransac.hpp"
#include "SfmUtils.hpp"
#include "legacy/geometry.hpp"
#include "degensac.h"
#include "legacy/geometry.hpp"

namespace rsfm
{
using Eigen::ArrayXf;

static constexpr int32_t maxNbSamplesForFHOpt = 128;

Eigen::ArrayX<bool> bernoulliSampleForFHOpt(Eigen::ArrayX<bool> const& mask, int32_t maskTrueCount, int32_t maxNbSamples) {
    assert(maskTrueCount == -1 || maskTrueCount == mask.count());
    auto const n0 = maskTrueCount == -1 ? cast32i(mask.count()) : maskTrueCount;
    bool const needSampling = n0 > maxNbSamples;
    Eigen::ArrayX<bool> tmpMask{};
    if (needSampling) {
        tmpMask.resizeLike(mask);
        std::mt19937_64 rng{static_cast<uint64_t>(mask.rows() * 11 + n0 * 7)};
        std::bernoulli_distribution dist{float(maxNbSamples) / n0};
        int32_t count;
        do {
            count = 0;
            for (int32_t i = 0; i < mask.size(); i++) {
                bool const selected = mask[i] && dist(rng);
                if (selected) {
                    count++;
                }
                tmpMask[i] = selected;
            }
        } while (count < maxNbSamples / 2);
    }
    return tmpMask;
}

inline std::pair<Eigen::Matrix3f, Eigen::ArrayX<bool>> optimiseEpipolarity(const std::array<ArrayXf, 4>& ptPairs, Eigen::ArrayX<bool> const& mask, float thres, int32_t maskTrueCount = -1)
{
    auto const tmpMask = bernoulliSampleForFHOpt(mask, maskTrueCount, maxNbSamplesForFHOpt);
    Matrix3f newEpipolarity = findEpipolarity(mask2pts(ptPairs, tmpMask.rows() != 0 ? tmpMask.data() : mask.data()), true);
    auto newMask = checkEpipolarTransformation(newEpipolarity, ptPairs, thres);
    return std::make_pair(newEpipolarity, std::move(newMask));
}

std::pair<int32_t, Eigen::ArrayX<bool>> checkEpipolarityAndOptimise(const std::array<ArrayXf, 4>& ptPairs, Eigen::Matrix3f& F, float thres)
{
    auto mask0 = checkEpipolarTransformation(F, ptPairs, thres);
    auto n0 = cast32i(mask0.count());
    if (n0 < 8) {
        return std::make_pair(n0, std::move(mask0));
    }
    auto [F1, mask1] = optimiseEpipolarity(ptPairs, mask0, thres, n0);
    auto n1 = cast32i(mask1.count());
    if (n0 < n1) {
        F = F1;
    }
    return n0 < n1 ? std::make_pair(n1, std::move(mask1)) : std::make_pair(n0, std::move(mask0));
}

inline std::pair<Eigen::Projective2f, Eigen::ArrayX<bool>> optimiseHomography(const std::array<ArrayXf, 4>& ptPairs, Eigen::ArrayX<bool> const& mask, float thres, int32_t maskTrueCount = -1)
{
    auto const tmpMask = bernoulliSampleForFHOpt(mask, maskTrueCount, maxNbSamplesForFHOpt);
    Eigen::Projective2f newH = findHomography(mask2pts(ptPairs, tmpMask.rows() != 0 ? tmpMask.data() : mask.data()), true);
    auto newMask = checkProjectiveTransformation(newH, ptPairs, thres);
    return std::make_pair(newH, std::move(newMask));
}
std::pair<int32_t, Eigen::ArrayX<bool>> checkHomographyAndOptimise(const std::array<ArrayXf, 4>& ptPairs, Eigen::Projective2f& H, float thres)
{
    auto mask0 = checkProjectiveTransformation(H, ptPairs, thres);
    auto n0 = cast32i(mask0.count());
    if (n0 < 4) {
        return std::make_pair(n0, std::move(mask0));
    }
    auto [H1, mask1] = optimiseHomography(ptPairs, mask0, thres, n0);
    auto n1 = cast32i(mask1.count());
    if (n0 < n1) {
        H = H1;
    }
    return n0 < n1 ? std::make_pair(n1, std::move(mask1)) : std::make_pair(n0, std::move(mask0));
}

DegensacSolution findEpipolarityDegensac(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim,
        float thresRelativeF, float thresRelativeH,
        float requiredRansacConfidence)
{
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	int32_t const nbTiePts = tiePtPairs[0].size();
    int32_t const nbPts = ptPairs[0].size();
    if (nbPts + nbTiePts < 7)
    {
        return {};
    }
    ASSERT(nbTiePts == 0 && "Tie points are not yet supported by DEGENSAC.");

    NonDuplicateSampler::Sampler<uint32_t, 7> sampler(nbPts, std::random_device{}(), 0, 1);
    float const thresF = thresRelativeF * sqrtf(firstImgDim * secondImgDim);
    float const thresH = thresRelativeH * sqrtf(firstImgDim * secondImgDim);

    DegensacSolution best{};
    assert(best.nbInliersF == 0 && best.nbInliersH == 0);
    int32_t maxNbIters = 10000;

    for (int32_t idxIter = 0; idxIter < maxNbIters; idxIter++)
    {
        auto const indices = sampler();
        Eigen::Matrix<float, 7, 4> const samples = sampleWithIndices(ptPairs, indices);
        Eigen::Matrix3f candidateF[3];
        uint32_t const nbCandidateF = findEpipolarityWith7Points(samples, candidateF);
        if (nbCandidateF == 0) {
            idxIter--;
            continue;
        }
        int32_t nbInliersF = 0;
        Eigen::Array<bool, Eigen::Dynamic, 1> maskF;
        Eigen::Matrix3f F = candidateF[0];
        for (uint32_t i = 0; i < nbCandidateF; i++) {
            auto [n, mask] = checkEpipolarityAndOptimise(ptPairs, candidateF[i], thresF);
            assert(n == mask.count());
            if (n > nbInliersF) {
                nbInliersF = n;
                F = candidateF[i];
                maskF.swap(mask);
            }
        }
        if (nbInliersF > best.nbInliersF) {
            best.nbInliersF = nbInliersF;
            best.F = F;
            constexpr int32_t nbDegTests = 5;
            static constexpr std::array<std::array<int32_t, 3>, nbDegTests> triplets = {
                0, 1, 2,
                3, 4, 5,
                0, 1, 6,
                3, 4, 6,
                2, 5, 6
            };
            static constexpr std::array<std::array<int32_t, 4>, nbDegTests> remainings = {
                3, 4, 5, 6,
                0, 1, 2, 6,
                2, 3, 4, 5,
                0, 1, 2, 5,
                0, 1, 3, 4
            };
            int32_t nbPlanarSamples = 3;
            Eigen::Projective2f H;
            for (int32_t i = 0; i < nbDegTests; i++) {
                auto const& tri = triplets.at(i);
                auto const& rem = remainings.at(i);
                Eigen::Matrix<float, 3, 4> triplet;
                triplet << samples.row(tri[0]), samples.row(tri[1]), samples.row(tri[2]);
                Eigen::Projective2f const candH  = findHomographyFromEpipolarity(F, triplet);
                Eigen::Matrix<float, 4, 4> remain;
                remain << samples.row(rem[0]), samples.row(rem[1]), samples.row(rem[2]), samples.row(rem[3]);
                thread_local std::array<Eigen::ArrayXf, 4> remPtPairs;
                for (int32_t j = 0; j < 4; j++) {
                    remPtPairs.at(j) = remain.col(j);
                }
                auto const mask = checkProjectiveTransformation(candH, remPtPairs, thresH);
                auto const n = mask.count() + 3;
                if (n > nbPlanarSamples) {
                    nbPlanarSamples = n;
                    H = candH;
                    // break; // Shall we break?
                }
            }
            if (nbPlanarSamples >= 5) {
                auto [nbInliersH, maskH] = checkHomographyAndOptimise(ptPairs, H, thresH);
                assert(nbInliersH == maskH.count());
                if (nbInliersH > best.nbInliersH) {
                    best.nbInliersH = nbInliersH;
                    best.H = H;
                }
                std::vector<uint32_t> nonPlanarPtIndices;
                nonPlanarPtIndices.reserve(nbPts);
                for (int32_t i = 0; i < nbPts; i++) {
                    if (!maskH[i]) {
                        nonPlanarPtIndices.push_back(i);
                    }
                }
                if (nonPlanarPtIndices.size() >= 2) {
                    std::array<ArrayXf, 4> nonPlanarPtPairs;
                    for (auto& x : nonPlanarPtPairs) {
                        x.resize(nonPlanarPtIndices.size());
                    }
                    for (uint32_t i = 0; i < nonPlanarPtIndices.size(); i++) {
                        for (int j = 0; j < 4; j++) {
                            nonPlanarPtPairs.at(j)[i] = ptPairs.at(j)[nonPlanarPtIndices[i]];
                        }
                    }
                    auto getSampleF = [&H, &nonPlanarPtPairs](std::array<uint32_t, 2> indices){
                        auto const samplePts = sampleWithIndices(nonPlanarPtPairs, indices);
                        return findEpipolarityFromHomography(H, samplePts);
                    };
                    auto ransacTest = [&getSampleF, &nonPlanarPtPairs, thresF](std::array<uint32_t, 2> indices) -> size_t {
                        auto const Fh = getSampleF(indices);
                        return checkEpipolarTransformation(Fh, nonPlanarPtPairs, thresF).count();
                    };
                    std::array<uint32_t, 2> const bestIndices = ransac<decltype(ransacTest), 2, uint32_t>(ransacTest, nonPlanarPtIndices.size(), requiredRansacConfidence);
                    auto Fh = getSampleF(bestIndices);
                    auto const [nbInliersFh, maskFh] = checkEpipolarityAndOptimise(ptPairs, Fh, thresF);
                    assert(nbInliersFh == maskFh.count());
                    if (nbInliersFh > best.nbInliersF) {
                        best.nbInliersF = nbInliersFh;
                        best.F = Fh;
                    }
                }
            }
            maxNbIters = cast32i(std::min<double>(maxNbIters, std::ceil(log2(1.0 - requiredRansacConfidence) / log2(1.0 - std::pow(double(best.nbInliersF) / nbPts, 7.0)))));
        }
    }
    auto const maskBestF = checkEpipolarTransformation(best.F, ptPairs, thresF);
    best.inlierMaskF = std::vector<bool>(maskBestF.begin(), maskBestF.end());
    if (best.nbInliersH != 0) {
        auto const maskBestH = checkProjectiveTransformation(best.H, ptPairs, thresH);
        best.inlierMaskH = std::vector<bool>(maskBestH.begin(), maskBestH.end());
    }
    return best;
}

// Using pre-computed homography from co-planar tie-points.
DegensacSolution findEpipolarityDegensacWithHomography(
        const std::array<ArrayXf, 4>& ptPairs, // only include outliers for H
        Matrix3f const& H,
        const std::array<ArrayXf, 4>& tiePtPairs, // only include outliers for H
        const int firstImgDim, const int secondImgDim,
        float thresRelativeF,
        float requiredRansacConfidence)
{
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	int32_t const nbTiePts = tiePtPairs[0].size();
    int32_t const nbPts = ptPairs[0].size();
    if (nbPts + nbTiePts < 3)
    {
        return {};
    }
    unused(ptPairs, H, tiePtPairs, firstImgDim, secondImgDim, thresRelativeF, requiredRansacConfidence);
    throwNotImplemented();
}

}