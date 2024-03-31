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

#include "ImagePair.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "RapidSift.h"
#include "legacy/geometry.hpp"
#include "ransac.hpp"
#include "Types.hpp"
#include "Builder.h"
#include "Image.h"
#include "DefaultCacheableObject.h"
#include "Config.h"
#include "Types.h"
#include "legacy/rt_optimiser.hpp"
#include "FiberUtils.h"
#include <cstdio>
#include "SfmUtils.h"
#include "SfmUtils.hpp"
#include "distortion.h"
#include <cmath>
#include <array>
#include "degensac.h"

#define DEBUG_PRINT 0

namespace rsfm
{
using Eigen::Matrix3f;
using Eigen::Projective2f;
using Eigen::ArrayXf;

namespace
{

std::array<ArrayXf, 4> makePtPair(const KeyPoint first[], const KeyPoint second[], const std::vector<Pair<Index>>& matches)
{
    std::array<ArrayXf, 4> result;
    for (auto& x : result){
        x.resize(static_cast<int64_t>(matches.size()));
    }
    for (unsigned i = 0; i < matches.size(); i++){
        const auto& m = matches[i];
        const auto& p0 = first[m.first].location;
        const auto& p1 = second[m.second].location;
        result[0][i] = p0.x;
        result[1][i] = p0.y;
        result[2][i] = p1.x;
        result[3][i] = p1.y;
    }
    return result;
}

std::array<ArrayXf, 4> makeTiePtPair(const TiePtMeasurementExt first[], const TiePtMeasurementExt second[], const std::vector<Pair<Index>>& matches)
{
    std::array<ArrayXf, 4> result;
    for (auto& x : result){
        x.resize(static_cast<int64_t>(matches.size()));
    }
    for (unsigned i = 0; i < matches.size(); i++){
        const auto& m = matches[i];
        const auto& p0 = first[m.first];
        const auto& p1 = second[m.second];
        result[0][i] = p0.x;
        result[1][i] = p0.y;
        result[2][i] = p1.x;
        result[3][i] = p1.y;
    }
    return result;
}

template <bool isTiePt>
std::array<ArrayXf, 4> makePtPairImpl(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches)
{
	if constexpr (isTiePt) {
		return makeTiePtPair(first.tiePtMeasurements.data(), second.tiePtMeasurements.data(), matches);
	}
	else {
		const auto loc = cudapp::storage::StorageLocation::kSysMem;
		const cudaStream_t stream = builder->anyStream();
		const auto kptsHolderFirst = cudapp::storage::acquireMemory<const KeyPoint>(builder->storageManager(), first.keyPoints, loc, stream, false, true);
		const auto kptsHolderSecond = cudapp::storage::acquireMemory<const KeyPoint>(builder->storageManager(), second.keyPoints, loc, stream, false, true);
		cudapp::fiberSyncCudaStream(stream);

		return makePtPair(kptsHolderFirst.data(), kptsHolderSecond.data(), matches);
	}
}

} // unnamed namespace

template <bool isTiePt>
std::array<ArrayXf, 4> makePtPair(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, const RealCamera& camFirst, const RealCamera& camSecond)
{
    std::array<ArrayXf, 4> ptPairs = makePtPairImpl<isTiePt>(builder, first, second, matches);
    undistortInPlace(ptPairs[0].data(), ptPairs[1].data(), static_cast<uint32_t>(matches.size()), camFirst);
    undistortInPlace(ptPairs[2].data(), ptPairs[3].data(), static_cast<uint32_t>(matches.size()), camSecond);
    return ptPairs;
}
template std::array<ArrayXf, 4> makePtPair<true>(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, const RealCamera& camFirst, const RealCamera& camSecond);
template std::array<ArrayXf, 4> makePtPair<false>(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, const RealCamera& camFirst, const RealCamera& camSecond);

template <bool isTiePt>
std::array<ArrayXf, 4> makePtPair(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, bool undistort)
{
    if (undistort) {
        return makePtPair<isTiePt>(builder, first, second, matches, *builder->getRealCamera(first.hCamera), *builder->getRealCamera(second.hCamera));
    }
    else {
        return makePtPairImpl<isTiePt>(builder, first, second, matches);
    }
}
template std::array<ArrayXf, 4> makePtPair<true>(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, bool undistort);
template std::array<ArrayXf, 4> makePtPair<false>(Builder* builder, const Image& first, const Image& second, const std::vector<Pair<Index>>& matches, bool undistort);

std::array<ArrayXf, 4> filterPtPair(const std::array<ArrayXf, 4>& orig, const std::vector<bool>& mask) {
    const auto nbInliers = countInliers(mask);
    std::array<ArrayXf, 4> result;
    for (auto& x : result) {
        x.resize(nbInliers, 1);
    }
    const auto nbOrigPairs = orig[0].rows();
    for (int i = 0, j = 0; i < nbOrigPairs; i++) {
        if (mask.at(i)) {
            for (int k = 0; k < 4; k++) {
                result[k][j] = orig[k][i];
            }
            j++;
        }
    }
    return result;
}

template <uint32_t nbTiePts, typename = std::enable_if_t<(nbTiePts < 8)>>
std::pair<Matrix3f, std::vector<bool>> findEpipolarityRansacWithInsufficientTiePts(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
	static_assert(nbTiePts < 8);
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	assert(tiePtPairs[0].size() == nbTiePts);

    //used to check the randomly sampled points are valid for estimation
    const float sanity_threshold[2] = {//use lower threshold than homography because we need 8 points and large threshold may be too strict
            square(firstImgDim * std::min(0.1f, thresRelative * 4)),
            square(secondImgDim * std::min(0.1f, thresRelative * 4))
    };

	constexpr uint32_t nbRandSamples = 8 - nbTiePts;

    auto ransacTest = [&](const std::array<uint32_t, nbRandSamples>& indices) -> size_t{
		constexpr auto nbIndices = nbRandSamples;
        Eigen::Matrix<float, nbIndices, 4> randPts = sampleWithIndices(ptPairs, indices);
        for(int i = 0; i < int(nbIndices); i++){
            for(int j = 0; j < i; j++){
                Eigen::Matrix<float, 1, 4> diff = randPts.row(i) - randPts.row(j);
                if(diff.template leftCols<2>().squaredNorm() < sanity_threshold[0]
                   || diff.template rightCols<2>().squaredNorm() < sanity_threshold[1])
                    return 0u;
            }
        }

		assert(nbIndices + tiePtPairs[0].size() == 8);
		const Eigen::Matrix<float, 8, 4> pts = concatSamples<8, nbRandSamples, 4>(randPts, tiePtPairs);

        const Eigen::Matrix3f trans = findEpipolarity(pts, preferAccuracy);

        return static_cast<size_t>(checkEpipolarTransformation(trans, ptPairs, firstImgDim * thresRelative).count());
    };

    std::array<uint32_t, nbRandSamples> bestIndices = ransac<decltype(ransacTest), nbRandSamples, uint32_t>(ransacTest, static_cast<size_t>(ptPairs[0].rows()), requiredRansacConfidence);
    Matrix3f epipolarity = findEpipolarity(concatSamples<8, nbRandSamples, 4>(sampleWithIndices(ptPairs, bestIndices), tiePtPairs), preferAccuracy);
    auto mask = checkEpipolarTransformation(epipolarity, ptPairs, firstImgDim * thresRelative);

    auto ransacNbInliners = mask.count();
    if (ransacNbInliners >= nbRandSamples) {
        Matrix3f newEpipolarity = findEpipolarity(concatSamples<Eigen::Dynamic, Eigen::Dynamic, 4>(mask2pts(ptPairs, mask.data()), tiePtPairs), preferAccuracy);
        auto newMask = checkEpipolarTransformation(newEpipolarity, ptPairs, firstImgDim * thresRelative);
        const auto recomputedNbInliers = newMask.count();
        if (recomputedNbInliers > ransacNbInliners) {
            epipolarity = newEpipolarity;
            mask = newMask;
            ransacNbInliners = recomputedNbInliers;
        }
    }
    std::vector<bool> resultMask(mask.data(), mask.data() + mask.rows());
    return std::make_pair(epipolarity, std::move(resultMask));
}

template <uint32_t nbTiePtSamples>
std::array<uint32_t, nbTiePtSamples> findBestTiePtSamples(const std::array<ArrayXf, 4>& tiePtPairs) {
	static_assert(nbTiePtSamples > 2);
	const uint32_t nbTiePts = cast32u(tiePtPairs[0].size());
	ASSERT(nbTiePtSamples <= nbTiePts);
	std::array<uint32_t, nbTiePtSamples> ret{};
	if (nbTiePts == nbTiePtSamples) {
		std::iota(ret.begin(), ret.end(), 0u);
		return ret;
	}
	auto getFirst = [&tiePtPairs](uint32_t i) {return Vec2f{tiePtPairs[0][i], tiePtPairs[1][i]};};
	auto getSecond = [&tiePtPairs](uint32_t i) {return Vec2f{tiePtPairs[2][i], tiePtPairs[3][i]};};
	auto getRow = [&tiePtPairs](uint32_t i) {return (Eigen::Array<float, 1, 4>{} << tiePtPairs[0][i], tiePtPairs[1][i], tiePtPairs[2][i], tiePtPairs[3][i]).finished(); };
	std::vector<bool> picked(size_t{nbTiePts}, false);
	Eigen::Matrix<float, nbTiePtSamples, 4> pickedPts;
	// find the point most far away from center as seed
	if constexpr (nbTiePtSamples > 0) {
		uint32_t& bestIdx = ret[0];
		float maxDist = 0.f;
		const Pair<Vec2f> c = {{tiePtPairs[0].mean(), tiePtPairs[1].mean()}, {tiePtPairs[2].mean(), tiePtPairs[3].mean()}};
		for (uint32_t i = 0; i < nbTiePts; i++) {
			const float dist = (getFirst(i) - c.first).squaredNorm() + (getSecond(i) - c.second).squaredNorm();
			if (dist > maxDist) {
				bestIdx = i;
				maxDist = dist;
			}
		}
		picked[bestIdx] = true;
		pickedPts.row(0) = getRow(bestIdx);
	}
	// The next is the point most far away from seed
	if constexpr (nbTiePtSamples > 1) {
		const uint32_t idxSeed = ret[0];
		const Pair<Vec2f> seed{getFirst(idxSeed), getSecond(idxSeed)};
		uint32_t& bestIdx = ret[1];
		float maxDist = 0;
		for (uint32_t i = 0; i < nbTiePts; i++) {
			if (picked[i]) {
				continue;
			}
			const float dist = (getFirst(i) - seed.first).squaredNorm() + (getSecond(i) - seed.second).squaredNorm();
			if (dist > maxDist) {
				bestIdx = i;
				maxDist = dist;
			}
		}
		picked[bestIdx] = true;
		pickedPts.row(1) = getRow(bestIdx);
	}
	// The rest is for max covariance determinant
	auto pick = [&]<uint32_t idx>() {
		if constexpr (nbTiePtSamples > idx) {
			uint32_t& bestIdx = ret[idx];
			float maxSpanFactor = 0;
			std::pair<Eigen::Matrix<float, idx + 1, 2>, Eigen::Matrix<float, idx + 1, 2>> pts;
			pts.first.template topRows<idx>() = pickedPts.template block<idx, 2>(0, 0);
			pts.second.template topRows<idx>() = pickedPts.template block<idx, 2>(0, 2);
			for (uint32_t i = 0; i < nbTiePts; i++) {
				if (picked[i]) {
					continue;
				}
				pts.first.row(idx) = toEigen(getFirst(i)).transpose();
				pts.second.row(idx) = toEigen(getSecond(i)).transpose();
				const auto spanFactor = computeCovariance(pts.first).determinant() * computeCovariance(pts.second).determinant();
				if (spanFactor > maxSpanFactor) {
					maxSpanFactor = spanFactor;
					bestIdx = i;
				}
			}
			picked[bestIdx] = true;
			pickedPts.row(idx) = getRow(bestIdx);
		}
	};
	pick.template operator()<2>();
	pick.template operator()<3>();
	pick.template operator()<4>();
	pick.template operator()<5>();
	pick.template operator()<6>();
	pick.template operator()<7>();
	static_assert(nbTiePtSamples <= 8, "template not instantiated for this");
	return ret;
}

std::pair<Matrix3f, std::vector<bool>> findEpipolarityWithSufficientTiePts(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
	unused(secondImgDim, requiredRansacConfidence);
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	const auto nbTiePts = tiePtPairs[0].size();
	ASSERT(nbTiePts >= 8);

    std::array<uint32_t, 8> bestIndices = findBestTiePtSamples<8>(tiePtPairs); // of tie points
    Matrix3f epipolarity = findEpipolarity(sampleWithIndices(tiePtPairs, bestIndices), preferAccuracy);
    auto mask = checkEpipolarTransformation(epipolarity, ptPairs, firstImgDim * thresRelative);

    auto ransacNbInliners = mask.count();
    if (ransacNbInliners >= 8) {
        Matrix3f newEpipolarity = findEpipolarity(concatSamples<Eigen::Dynamic, Eigen::Dynamic, 4>(mask2pts(ptPairs, mask.data()), tiePtPairs), preferAccuracy);
        auto newMask = checkEpipolarTransformation(newEpipolarity, ptPairs, firstImgDim * thresRelative);
        const auto recomputedNbInliers = newMask.count();
        if (recomputedNbInliers > ransacNbInliners) {
            epipolarity = newEpipolarity;
            mask = newMask;
            ransacNbInliners = recomputedNbInliers;
        }
    }
    std::vector<bool> resultMask(mask.data(), mask.data() + mask.rows());
    return std::make_pair(epipolarity, std::move(resultMask));
}

std::pair<Matrix3f, std::vector<bool>> findEpipolarityRansac(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	const auto nbTiePts = tiePtPairs[0].size();

    if (nbTiePts < 8) {
		switch (nbTiePts) {
			case 0: return findEpipolarityRansacWithInsufficientTiePts<0>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 1: return findEpipolarityRansacWithInsufficientTiePts<1>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 2: return findEpipolarityRansacWithInsufficientTiePts<2>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 3: return findEpipolarityRansacWithInsufficientTiePts<3>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 4: return findEpipolarityRansacWithInsufficientTiePts<4>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 5: return findEpipolarityRansacWithInsufficientTiePts<5>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 6: return findEpipolarityRansacWithInsufficientTiePts<6>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			case 7: return findEpipolarityRansacWithInsufficientTiePts<7>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
			default: DIE("You should never reach here");
		}
	}
	else {
		const auto results = findEpipolarityWithSufficientTiePts(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
		// It may fail when tie points are in a plane, so also try without tie points
		const auto baseline = findEpipolarityRansacWithInsufficientTiePts<0>(ptPairs, {}, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
		return countInliers(results.second) >= countInliers(baseline.second) ? std::move(results) : std::move(baseline);
	}
}

template <uint32_t nbTiePts>
std::pair<Eigen::Projective2f, std::vector<bool>> findHomographyRansacWithInsufficientTiePts(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
	static_assert(nbTiePts < 4);
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	assert(tiePtPairs[0].size() == nbTiePts);

    //used to check the randomly sampled points are valid for estimation
    const float sanity_threshold[2] = {
        square(firstImgDim * std::min(0.25f, thresRelative * 8)),
        square(secondImgDim * std::min(0.25f, thresRelative * 8))
        };

	constexpr uint32_t nbRandSamples = 4 - nbTiePts;

    auto ransac_test = [&](const std::array<uint32_t, nbRandSamples>& indices) -> size_t{
        Eigen::Matrix<float, nbRandSamples, 4> randPts = sampleWithIndices(ptPairs, indices);
        for(int i = 0; i < int(indices.size()); i++){
            for(int j = 0; j < i; j++){
                Eigen::Matrix<float, 1, 4> diff = randPts.row(i) - randPts.row(j);
                if(diff.template leftCols<2>().squaredNorm() < sanity_threshold[0]
                        || diff.template rightCols<2>().squaredNorm() < sanity_threshold[1])
                    return 0u;
            }
        }

		const Eigen::Matrix<float, 4, 4> pts = concatSamples<4, nbRandSamples, 4>(randPts, tiePtPairs);
        const Eigen::Projective2f trans = findHomography(pts, preferAccuracy);

        const auto nbInliners = static_cast<size_t>(checkProjectiveTransformation(trans, ptPairs, firstImgDim * thresRelative).count());
        return nbInliners;
    };

    const std::array<uint32_t, nbRandSamples> best_indices = ransac<decltype(ransac_test), nbRandSamples, uint32_t>(ransac_test, static_cast<size_t>(ptPairs[0].rows()), requiredRansacConfidence);
    Projective2f homography = findHomography(concatSamples<4, nbRandSamples, 4>(sampleWithIndices(ptPairs, best_indices), tiePtPairs), preferAccuracy);
    auto mask = checkProjectiveTransformation(homography, ptPairs, firstImgDim * thresRelative);

    auto ransacNbInliners = mask.count();
    if (ransacNbInliners > 4) {
        // @fixme: this almost always result in worse homography.
        // Maybe because these points are usually not in the same plane?
        const Projective2f newHomography = findHomography(concatSamples<Eigen::Dynamic, Eigen::Dynamic, 4>(mask2pts(ptPairs, mask.data()), tiePtPairs), preferAccuracy);
        const auto newMask = checkProjectiveTransformation(newHomography, ptPairs, firstImgDim * thresRelative);
        const auto recomputedNbInliers = newMask.count();
        if (recomputedNbInliers > ransacNbInliners) {
            homography = newHomography;
            mask = newMask;
#ifndef DEBUG_PRINT
            printf("Info: recomputed H number of inliers %ld -> %ld\n", ransacNbInliners, recomputedNbInliers);
#endif
            ransacNbInliners = recomputedNbInliers;
        }
#ifndef DEBUG_PRINT
        else {
            printf("Warning: recomputed H number of inliers %ld -> %ld\n", ransacNbInliners, recomputedNbInliers);
        }
#endif
    }
    std::vector<bool> resultMask(mask.data(), mask.data() + mask.rows());

    return std::make_pair(homography, std::move(resultMask));
}

std::pair<Eigen::Projective2f, std::vector<bool>> findHomographyWithSufficientTiePts(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
	unused(secondImgDim, requiredRansacConfidence);
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	const auto nbTiePts = tiePtPairs[0].size();
	ASSERT(nbTiePts >= 4);

    std::array<uint32_t, 4> bestIndices = findBestTiePtSamples<4>(tiePtPairs); // of tie points
    Projective2f homography = findHomography(sampleWithIndices(ptPairs, bestIndices), preferAccuracy);
    auto mask = checkProjectiveTransformation(homography, ptPairs, firstImgDim * thresRelative);

    auto ransacNbInliners = mask.count();
    if (ransacNbInliners > 4) {
        // @fixme: this almost always result in worse homography.
        // Maybe because these points are usually not in the same plane?
        const Projective2f newHomography = findHomography(concatSamples<Eigen::Dynamic, Eigen::Dynamic, 4>(mask2pts(ptPairs, mask.data()), tiePtPairs), preferAccuracy);
        const auto newMask = checkProjectiveTransformation(newHomography, ptPairs, firstImgDim * thresRelative);
        const auto recomputedNbInliers = newMask.count();
        if (recomputedNbInliers > ransacNbInliners) {
            homography = newHomography;
            mask = newMask;
#ifndef DEBUG_PRINT
            printf("Info: recomputed H number of inliers %ld -> %ld\n", ransacNbInliners, recomputedNbInliers);
#endif
            ransacNbInliners = recomputedNbInliers;
        }
#ifndef DEBUG_PRINT
        else {
            printf("Warning: recomputed H number of inliers %ld -> %ld\n", ransacNbInliners, recomputedNbInliers);
        }
#endif
    }
    std::vector<bool> resultMask(mask.data(), mask.data() + mask.rows());

    return std::make_pair(homography, std::move(resultMask));
}

std::pair<Eigen::Projective2f, std::vector<bool>> findHomographyRansac(
        const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
        const int firstImgDim, const int secondImgDim, const float thresRelative,
        bool preferAccuracy, float requiredRansacConfidence)
{
    assert(allEqual(ptPairs[0].rows(), ptPairs[1].rows(), ptPairs[2].rows(), ptPairs[3].rows()));
	assert(allEqual(tiePtPairs[0].rows(), tiePtPairs[1].rows(), tiePtPairs[2].rows(), tiePtPairs[3].rows()));
	const auto nbTiePts = tiePtPairs[0].size();

	// It may fail if tie points are not in a plane, so also try without tie points
	const auto baseline = findHomographyRansacWithInsufficientTiePts<0>(ptPairs, {}, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
	if (nbTiePts == 0) {
		return baseline;
	}

	const auto results = [&]{
		if (nbTiePts < 4) {
			switch (nbTiePts) {
				case 0: DIE("You should never reach here");
				case 1: return findHomographyRansacWithInsufficientTiePts<1>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
				case 2: return findHomographyRansacWithInsufficientTiePts<2>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
				case 3: return findHomographyRansacWithInsufficientTiePts<3>(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
				default: DIE("You should never reach here");
			}
		}
		else {
			return findHomographyWithSufficientTiePts(ptPairs, tiePtPairs, firstImgDim, secondImgDim, thresRelative, preferAccuracy, requiredRansacConfidence);
		}
	}();

	return countInliers(results.second) >= countInliers(baseline.second) ? std::move(results) : std::move(baseline);
}

std::pair<std::vector<uint32_t>, Eigen::MatrixX4f> gatherInliers(const std::array<ArrayXf, 4>& ptPairs,
    const std::vector<bool>& initInlierMask)
{
    auto initInlierIndices = mask2indices(initInlierMask);
    assert(static_cast<int>(initInlierIndices.size()) == countInliers(initInlierMask));
    Eigen::MatrixX4f initInlierPtPairs{initInlierIndices.size(), 4};
    for (unsigned i = 0; i < initInlierIndices.size(); i++)
    {
        const auto idx = initInlierIndices[i];
        initInlierPtPairs.row(i) << ptPairs[0][idx], ptPairs[1][idx], ptPairs[2][idx], ptPairs[3][idx];
    }
    return {std::move(initInlierIndices), std::move(initInlierPtPairs)};
}

std::vector<bool> recheckInliers(const std::array<ArrayXf, 4>& ptPairs,
    const Eigen::Isometry3f& rt, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, const float threshold, bool checkIsFront, Eigen::MatrixX3f const* pPts3dFirst = nullptr) // camera-point-camera angle
{
    REQUIRE(threshold >= 0);
    bool const checkError = std::isfinite(threshold);
    REQUIRE(checkError || checkIsFront);
    REQUIRE(!pPts3dFirst || ptPairs.at(0).rows() == pPts3dFirst->rows());

    const Eigen::MatrixX3f& pts3dFirst = pPts3dFirst ? *pPts3dFirst : triangulate(firstCam, secondCam, rt, ptPairs); // @fixme: use rt_optimizer to refine pts3d and minimize projection error

    auto cvtRet = [](Eigen::Array<bool, Eigen::Dynamic, 1> const& validMask){
        return std::vector<bool>(validMask.data(), validMask.data() + validMask.rows());
    };
    auto getValidMaskFront = [&]() {
        REQUIRE(checkIsFront);
        const Eigen::Array<bool, Eigen::Dynamic, 1> validMask = (pts3dFirst.col(2).array() > 0) && ((pts3dFirst.rowwise().homogeneous() * rt.matrix().row(2).transpose().eval()).array() > 0);
        return validMask;
    };
    auto getValidMaskError = [&]() {
        REQUIRE(checkError);
        const Eigen::MatrixX2f pts2dFirst = (toKMat<float>(firstCam) * pts3dFirst.transpose()).colwise().hnormalized().transpose(); // @fixme: this is slow
        const Eigen::MatrixX2f pts2dSecond = (pts3dFirst.rowwise().homogeneous() * (toKMat<float>(secondCam) * rt.matrix().template topRows<3>()).transpose().eval()).rowwise().hnormalized(); // @fixme: this is slow
        const Eigen::Array<bool, Eigen::Dynamic, 1> validMask = (((pts2dFirst.col(0).array() - ptPairs[0]).square() + (pts2dFirst.col(1).array() - ptPairs[1]).square())
            * ((pts2dSecond.col(0).array() - ptPairs[2]).square() + (pts2dSecond.col(1).array() - ptPairs[3]).square()) < std::pow(threshold, 4));
        return validMask;
    };
    if (checkError && checkIsFront)
        return cvtRet(getValidMaskError() && getValidMaskFront());
    else if (checkError)
        return cvtRet(getValidMaskError());
    else
        return cvtRet(getValidMaskFront());
}

std::pair<float, float> getMedianZAndAngleCPC(
    const Eigen::Isometry3f& rt, Eigen::MatrixX3f const& pts3d, std::vector<uint32_t> const& inliers, uint64_t seed)
{
    std::mt19937_64 rng{seed * 23 + 7};
    constexpr uint32_t N = 128;
    std::array<uint32_t, N> samples{};
    size_t const nbSamples = std::sample(inliers.begin(), inliers.end(), samples.begin(), N, rng) - samples.begin();
    std::array<float, N> depth{};
    std::array<float, N> angle{};
    Eigen::Vector3f const c2 = toEigen(fromEigen(rt).toPose().C);
    for (uint32_t i = 0; i < nbSamples; i++) {
        uint32_t const s = samples[i];
        Eigen::Vector3f const p = pts3d.row(s).transpose();
        depth[i] = std::sqrt(p[2] * (rt*p).eval()[2]);
        Eigen::Vector3f const v2 = p-c2;
        angle[i] = std::acos(p.dot(v2) / std::sqrt(p.squaredNorm() * v2.squaredNorm()));
    }
    std::nth_element(depth.begin(), depth.begin() + nbSamples/2, depth.begin() + nbSamples);
    std::nth_element(angle.begin(), angle.begin() + nbSamples/2, angle.begin() + nbSamples);
    return std::make_pair(depth.at(nbSamples/2), angle.at(nbSamples/2));
}
#if 0
std::vector<bool> recheckInliers(const std::array<ArrayXf, 4>& ptPairs,
    const Eigen::Isometry3f& rt, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam,
    const float zMin, const float zMax, const float angleMin, const float angleMax, const float threshold)
{
    REQUIRE(zMin < zMax);
    REQUIRE(angleMin < angleMax);
    const Eigen::MatrixX3f pts3dFirst = triangulate(firstCam, secondCam, rt, ptPairs); // @fixme: use rt_optimizer to refine pts3d and minimize projection error
    const Eigen::VectorXf depthSecond = pts3dFirst.rowwise().homogeneous() * rt.matrix().row(2).transpose(); // @fixme: this is slow
    const Eigen::Array<bool, Eigen::Dynamic, 1> depthValid =
        std::isfinite(zMax)
        ?   ((pts3dFirst.col(2).array() > zMin)
            && (pts3dFirst.col(2).array() < zMax)
            && depthSecond.array() > zMin
            && depthSecond.array() < zMax).eval()
        :   ((pts3dFirst.col(2).array() > zMin)
            && depthSecond.array() > zMin).eval();
#ifndef DEBUG_PRINT
    printf("[Debug] depthValid: %d/%d\n", (int)depthValid.count(), (int)depthValid.rows());
#endif
    Eigen::Array<bool, Eigen::Dynamic, 1> angleValid;
    if (angleMin > 0 || angleMax < float(M_PI)) { // use logic OR because we want the check when either threshold is valid
        const Eigen::ArrayXf x = pts3dFirst.col(0);
        const Eigen::ArrayXf y = pts3dFirst.col(1);
        const Eigen::ArrayXf z = pts3dFirst.col(2);
        const Eigen::Vector3f c2 = toEigen(fromEigen(rt).toPose().C);
        const Eigen::ArrayXf dotProd = x * (x - c2[0]) + y * (y - c2[1]) + z * (z - c2[2]);
        const Eigen::ArrayXf lenProd = ((x.square() + y.square() + z.square()) * ((x - c2[0]).square() + (y - c2[1]).square() + (z - c2[2]).square())).sqrt();
        REQUIRE(angleMax <= float(M_PI));
        angleValid = (dotProd < std::cos(angleMin) * lenProd) && (dotProd > std::cos(angleMax) * lenProd);
    }
    else {
        angleValid.setConstant(ptPairs[0].rows(), 1, true);
    }
#ifndef DEBUG_PRINT
    printf("[Debug] angleValid: %d/%d\n", (int)angleValid.count(), (int)angleValid.rows());
#endif
    if (std::isfinite(threshold) && threshold > 0) {
        const Eigen::MatrixX2f pts2dFirst = (toKMat<float>(firstCam) * pts3dFirst.transpose()).colwise().hnormalized().transpose(); // @fixme: this is slow
        const Eigen::MatrixX2f pts2dSecond = (pts3dFirst.rowwise().homogeneous() * (toKMat<float>(secondCam) * rt.matrix().template topRows<3>()).transpose()).rowwise().hnormalized(); // @fixme: this is slow
        const Eigen::Array<bool, Eigen::Dynamic, 1> validMask = depthValid && angleValid
            && (((pts2dFirst.col(0).array() - ptPairs[0]).square() + (pts2dFirst.col(1).array() - ptPairs[1]).square())
            * ((pts2dSecond.col(0).array() - ptPairs[2]).square() + (pts2dSecond.col(1).array() - ptPairs[3]).square()) < std::pow(threshold, 4));
        assert(validMask.rows() == ptPairs[0].rows());
        return std::vector<bool>(validMask.data(), validMask.data() + validMask.rows());
    }
    else {
        const Eigen::Array<bool, Eigen::Dynamic, 1> validMask = depthValid && angleValid;
        return std::vector<bool>(validMask.data(), validMask.data() + validMask.rows());
    }
}
#endif
std::pair<Isometry3f, std::vector<bool>> refineSolution(const Isometry3f& initRT,
	const std::array<ArrayXf, 4>& ptPairs, const std::vector<bool>& initInlierMask,
	const std::array<ArrayXf, 4>& tiePtPairs, float tiePtWeight,
    const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, float huber = INFINITY)
{
	const uint32_t nbTiePts = cast32u(tiePtPairs[0].size());
    auto [initInlierIndices, initInlierPtPairs] = gatherInliers(ptPairs, initInlierMask);
#if DEBUG_PRINT
    printf("%u points involved in RT optimization\n", (unsigned)initInlierPtPairs.rows());
#endif
    if (initInlierIndices.size() + nbTiePts < 8U){
        std::vector<bool> result(initInlierMask.size());
        result.assign(result.size(), false);
#if DEBUG_PRINT
        printf("insufficient points for RT optimization\n");
#endif
        return {initRT, std::move(result)};
    }

    thread_local static legacy::rt_optimiser optimizer;
    optimizer.set_camera(firstCam, secondCam);
	const bool useHuber = (std::isfinite(huber) && huber > 0);
	const bool useWeights = (nbTiePts != 0);
	Eigen::VectorXf huberVec[2];
	Eigen::VectorXf weightVec[2];
	auto setOptimizerInput = [nbTiePts, &tiePtPairs, tiePtWeight, huber, useHuber, useWeights, &huberVec, &weightVec]
		(const Isometry3f& initRT_, const Eigen::Matrix<float, Eigen::Dynamic, 4>& initInlierPtPairs_) mutable
	{
		optimizer.set_input(initRT_, concatSamples<Eigen::Dynamic, Eigen::Dynamic, 4>(initInlierPtPairs_, tiePtPairs));
		const auto nbInliers = initInlierPtPairs_.rows();

		for (Eigen::VectorXf& h : huberVec) {
			if (useHuber) {
				h.resize(nbInliers + nbTiePts);
				h.topRows(nbInliers).array() = huber;
				h.bottomRows(nbTiePts).array() = INFINITY;
			}
			else {
				h = Eigen::VectorXf{};
			}
		}
		for (Eigen::VectorXf& w : weightVec) {
			if (useWeights) {
				w.resize(nbInliers + nbTiePts);
				w.topRows(nbInliers).array() = 1.f;
				w.bottomRows(nbTiePts).array() = tiePtWeight;
			}
			else {
				w = Eigen::VectorXf{};
			}
		}
	};
    setOptimizerInput(initRT, initInlierPtPairs);

    using State = legacy::rt_optimiser::enum_state;
    State state = State::NumericFailure;
    for (int i = 0; i < 3; i++) {
#if DEBUG_PRINT
        const bool verbose = true;
#else
        const bool verbose = false;
#endif
        state = optimizer.optimise(verbose, huberVec[0], huberVec[1], weightVec[0], weightVec[1]);
        if (state != State::NumericFailure) {
            break;
        }
        else {
            const Isometry3f newRT = optimizer.get_RT();
            bool const checkIsFront = true; // @fixme: When issue #10 is fixed, set this to false.
            std::vector<bool> newMask = recheckInliers(ptPairs, newRT, firstCam, secondCam, INFINITY, checkIsFront);
            assert(newMask.size() == initInlierMask.size());
            for (unsigned i = 0; i < newMask.size(); i++) {
                newMask[i] = newMask[i] && initInlierMask[i];
            }
            std::tie(initInlierIndices, initInlierPtPairs) = gatherInliers(ptPairs, newMask);
            if (initInlierIndices.size() + nbTiePts < 8U){
                newMask.assign(newMask.size(), false);
                return {newRT, std::move(newMask)};
            }
			setOptimizerInput(newRT, initInlierPtPairs);
        }
    }
    if (state == State::NumericFailure) {
		// printf("[Warning] numeric failure\n"); // one known case is negative depth caused by incorrect homography solution.
        std::pair<Isometry3f, std::vector<bool>> result{initRT, {}};
        result.second.assign(initInlierMask.size(), false);
        return result;
    }

    const Isometry3f rt = optimizer.get_RT();
	const uint32_t nbNonTiePts = optimizer.getNbPts() - nbTiePts;
    auto l2InlierMask = optimizer.get_inlier_mask(huberVec[0].data(), huberVec[1].data());
	l2InlierMask.resize(nbNonTiePts);
    // const Eigen::MatrixX3f pts3d = optimizer.get_points().topRows(nbNonTiePts);
    std::vector<bool> inlierMask = initInlierMask;
    ASSERT(initInlierIndices.size() == l2InlierMask.size());
    for (unsigned i = 0; i < l2InlierMask.size(); i++) {
        assert(inlierMask[initInlierIndices[i]]);
        if (!l2InlierMask[i]){
            assert(initInlierIndices[i] < inlierMask.size());
            inlierMask[initInlierIndices[i]] = false;
        }
    }

    return {rt, std::move(inlierMask)};
}

#if 0
template <typename Transforms>
EigenAlignedVector<Isometry3f> filterDecomposedSolutions(const Transforms& transforms, const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tieptPairs, float tiePtWeight,
const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, const float zMin, const float zMax, const float angleMin, const float angleMax,
const float inlierThreshold, const uint32_t minNbInliers, const float disambiguityNbInliersRatio)
{
    static std::atomic_bool warningIsPrinted = false;
    if (!warningIsPrinted.load()) {
        printf("@fixme: we can't filter solutions like this. In case the correct solutions violates contraints for zMin/zMax/angleMin/angleMax while the wrong solution does not, it will keep the wrong solution!\n");
        warningIsPrinted.store(true);
    }
    EigenAlignedVector<std::pair<Isometry3f, int64_t>> solutions;
    //@info: may be parallelized with fiber (with mutex for solutions)
    for (const auto& rt : transforms) {
        // @fixme: use threshold=INFINITY to put all pairs in, and use Cauchy loss function instead of huber
#if 0
        const std::vector<bool> mask = recheckInliers(ptPairs, rt, firstCam, secondCam, zMin, zMax, angleMin, angleMax, inlierThreshold);
#else
        const std::vector<bool> mask = recheckInliers(ptPairs, rt, firstCam, secondCam, zMin, zMax, angleMin, angleMax, INFINITY);
#endif
//        printf("%ld/%lu\n", countInliers(mask), mask.size());
        if (countInliers(mask) >= minNbInliers) {
            const Isometry3f transform = refineSolution(rt, ptPairs, mask, tiePtPairs, tiePtWeight, firstCam, secondCam, inlierThreshold, zMin, zMax, angleMin, angleMax).first;
            const int64_t nbInliers = countInliers(recheckInliers(ptPairs, transform, firstCam, secondCam, zMin, zMax, angleMin, angleMax, inlierThreshold));
            solutions.emplace_back(transform, nbInliers);
        }
    }
    std::sort(solutions.begin(), solutions.end(), [](const auto& a, const auto& b){return a.second > b.second;});
    if (!solutions.empty())
    {
        const int64_t disambiguityMinNbInliers = int64_t(std::round(solutions.front().second * disambiguityNbInliersRatio));
        solutions.erase(std::find_if(solutions.begin(), solutions.end(), [disambiguityMinNbInliers](const auto& x){return x.second < disambiguityMinNbInliers;}), solutions.end());
    }
    EigenAlignedVector<Isometry3f> result(solutions.size());
    std::transform(solutions.begin(), solutions.end(), result.begin(), [](const auto& x) { return x.first; });
    return result;
}
#endif

std::vector<bool> isInFrontOfCam(const std::array<ArrayXf, 4>& ptPairs,
    const Eigen::Isometry3f& rt, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam)
{
    const Eigen::MatrixX3f pts3dFirst = triangulate(firstCam, secondCam, rt, ptPairs);
    const Eigen::VectorXf depthSecond = pts3dFirst.rowwise().homogeneous() * rt.matrix().row(2).transpose(); // @fixme: this is slow
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> mask = pts3dFirst.col(2).array() > 0.f && depthSecond.array() > 0.f;
    return std::vector<bool>(mask.data(), mask.data() + mask.size());
}

// remove invalid solution that has many points in the back of the camera.
template <typename Transforms>
EigenAlignedVector<std::pair<Isometry3f, int64_t>> disambiguateSolutions(const Transforms& transforms,
    const std::array<ArrayXf, 4>& ransacInlierPtPairs,
    const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, const float disambiguityNbInliersRatio)
{
    EigenAlignedVector<std::pair<Isometry3f, int64_t>> solutions;
    for (const auto& rt : transforms) {
        const std::vector<bool> mask = isInFrontOfCam(ransacInlierPtPairs, rt, firstCam, secondCam);
//        printf("%ld/%lu\n", countInliers(mask), mask.size());
        solutions.emplace_back(rt, countInliers(mask));
    }
    std::sort(solutions.begin(), solutions.end(), [](const auto& a, const auto& b){return a.second > b.second;});
    if (!solutions.empty())
    {
        const int64_t disambiguityMinNbInliers = int64_t(std::round(ransacInlierPtPairs[0].rows() * disambiguityNbInliersRatio));
        solutions.erase(std::find_if(solutions.begin(), solutions.end(), [disambiguityMinNbInliers](const auto& x){return x.second < disambiguityMinNbInliers || x.second < 2;}), solutions.end());
    }
    return solutions;
}

EigenAlignedVector<std::pair<Isometry3f, int64_t>> solveByEpipolarity(const Config& cfg, const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
	const int firstImgDim, const int secondImgDim, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam)
{
	if (ptPairs[0].size() < 8) {
		return {};
	}
    const auto [epipolarity, epipolarInlierMask] = findEpipolarityRansac(ptPairs, tiePtPairs, firstImgDim, secondImgDim, cfg.pairSolver.epipolarityRansacRelativeThreshold, cfg.pairSolver.preferAccuracy, cfg.pairSolver.requiredRansacConfidence);
    const auto epipolarTransforms = decomposeEpipolarity(firstCam, secondCam, epipolarity);
    return disambiguateSolutions(epipolarTransforms, filterPtPair(ptPairs, epipolarInlierMask), firstCam, secondCam, cfg.pairSolver.disambiguityFrontRatio);

    // const int minImgDim = std::min(firstImgDim, secondImgDim);
    // const float inlierThreshold = minImgDim * std::max(cfg.pairSolver.epipolarityRansacRelativeThreshold, cfg.pairSolver.recheckRelativeThreshold);
    // return filterDecomposedSolutions(epipolarTransforms, ptPairs, firstCam, secondCam, cfg.pairSolver.zMin, cfg.pairSolver.zMax, cfg.pairSolver.minAngle, cfg.pairSolver.maxAngle,
    //     inlierThreshold, cfg.pairSolver.minNbInliers, cfg.pairSolver.disambiguityNbInliersRatio);
}
EigenAlignedVector<std::pair<Isometry3f, int64_t>> solveByHomography(const Config& cfg, const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
	const int firstImgDim, const int secondImgDim, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam)
{
	if (ptPairs[0].size() < 4) {
		return {};
	}
    const auto [homography, homographyInlierMask] = findHomographyRansac(ptPairs, tiePtPairs, firstImgDim, secondImgDim, cfg.pairSolver.homographyRansacRelativeThreshold, cfg.pairSolver.preferAccuracy, cfg.pairSolver.requiredRansacConfidence);
    if (countInliers(homographyInlierMask) < cfg.pairSolver.minNbInliers) {
        return {};
    }
    const auto homographyDecomp = decomposeHomography(firstCam, secondCam, homography.matrix());
    EigenAlignedVector<Isometry3f> homographyTransforms(homographyDecomp.size());
    std::transform(homographyDecomp.begin(), homographyDecomp.end(), homographyTransforms.begin(), [](const HomographyFactors<float>& f){
        Isometry3f rt{f.R};
        rt.translation() = f.T_by_d.normalized();
        return rt;
    });

    const auto ransacInlierPtPairs = filterPtPair(ptPairs, homographyInlierMask);
    return disambiguateSolutions(homographyTransforms, ransacInlierPtPairs, firstCam, secondCam, cfg.pairSolver.disambiguityFrontRatio);

    // const int minImgDim = std::min(firstImgDim, secondImgDim);
    // const float inlierThreshold = minImgDim * std::max(cfg.pairSolver.homographyRansacRelativeThreshold, cfg.pairSolver.recheckRelativeThreshold);
    // return filterDecomposedSolutions(homographyTransforms, ptPairs, firstCam, secondCam, cfg.pairSolver.zMin, cfg.pairSolver.zMax, cfg.pairSolver.minAngle, cfg.pairSolver.maxAngle,
    //     inlierThreshold, cfg.pairSolver.minNbInliers, cfg.pairSolver.disambiguityNbInliersRatio);
}

// Both epipolarity and homography with degensac
std::pair<EigenAlignedVector<std::pair<Isometry3f, int64_t>>, EigenAlignedVector<std::pair<Isometry3f, int64_t>>>
solveByDegensacForBoth(const Config& cfg, const std::array<ArrayXf, 4>& ptPairs, const std::array<ArrayXf, 4>& tiePtPairs,
	const int firstImgDim, const int secondImgDim, const PinHoleCamera& firstCam, const PinHoleCamera& secondCam)
{
    if (ptPairs[0].size() < 7) {
        return {};
    }
    DegensacSolution const solution = findEpipolarityDegensac(ptPairs, tiePtPairs, firstImgDim, secondImgDim,
        cfg.pairSolver.epipolarityRansacRelativeThreshold,
        cfg.pairSolver.homographyRansacRelativeThreshold,
        cfg.pairSolver.requiredRansacConfidence);
#if DEBUG_PRINT
    printf("Degensac: nbInliersF = %d, nbInliersH = %d\n", solution.nbInliersF, solution.nbInliersH);
    assert(countInliers(solution.inlierMaskF) == solution.nbInliersF);
    assert(countInliers(solution.inlierMaskH) == solution.nbInliersH);
#endif
    EigenAlignedVector<std::pair<Isometry3f, int64_t>> solutionF;
    if (solution.nbInliersF >= (int32_t)cfg.pairSolver.minNbInliers) {
        const auto epipolarTransforms = decomposeEpipolarity(firstCam, secondCam, solution.F);
        // @fixme: if we don't get exactly one solution here, typically it's because of short baseline and RT error. See issue #4
        solutionF = disambiguateSolutions(epipolarTransforms, filterPtPair(ptPairs, solution.inlierMaskF), firstCam, secondCam, cfg.pairSolver.disambiguityFrontRatio);
#if DEBUG_PRINT
        printf("epipolarTransforms.size()=%d, solutionF.size()=%d\n", (int)epipolarTransforms.size(), (int)solutionF.size());
#endif
    }
    EigenAlignedVector<std::pair<Isometry3f, int64_t>> solutionH;
    if (solution.nbInliersH >= (int32_t)cfg.pairSolver.minNbInliers) {
        const auto homographyDecomp = decomposeHomography(firstCam, secondCam, solution.H.matrix());
        EigenAlignedVector<Isometry3f> homographyTransforms(homographyDecomp.size());
        std::transform(homographyDecomp.begin(), homographyDecomp.end(), homographyTransforms.begin(), [](const HomographyFactors<float>& f){
            Isometry3f rt{f.R};
            rt.translation() = f.T_by_d.normalized();
            return rt;
        });
        const auto ransacInlierPtPairs = filterPtPair(ptPairs, solution.inlierMaskH);
        solutionH = disambiguateSolutions(homographyTransforms, ransacInlierPtPairs, firstCam, secondCam, cfg.pairSolver.disambiguityFrontRatio);
#if DEBUG_PRINT
        printf("homographyDecomp.size()=%d, solutionH.size()=%d\n", (int)homographyDecomp.size(), (int)solutionH.size());
#endif
    }
    return std::make_pair(std::move(solutionF), std::move(solutionH));
}

static bool checkRotationClose(const Isometry3f& a, const Isometry3f& b, const float angleThreshold){
    return std::abs(Eigen::AngleAxis<float>{a.linear() * b.linear().transpose()}.angle()) < angleThreshold;
};

EigenAlignedVector<Isometry3f> removeDuplicateSolutions(const EigenAlignedVector<Isometry3f>& src, const float angleThreshold) {
    EigenAlignedVector<Isometry3f> result;
    for (unsigned i = 0; i < src.size(); i++){
        bool foundDuplicate = false;
        for (unsigned j = 0; j + 1 < i; j++){
            if (checkRotationClose(src[j], src[i], angleThreshold)) {
                foundDuplicate = true;
                break;
            }
        }
        if (!foundDuplicate){
            result.emplace_back(src[i]);
        }
    }
    return result;
}

float estimateScore(const Isometry3f& rt, const Eigen::MatrixX4f& inlierPtPairs, const int2 firstImgSize, const int2 secondImgSize,
const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, float relativeThreshold);

float solutionDiff(const Isometry3f& x, const Isometry3f& y) {
    const float rotFactor = 1.f;
    const float transFactor = 0.f;
    return std::abs(Eigen::AngleAxis<float>{x.linear() * y.linear().transpose()}.angle()) * rotFactor
         + std::acos(x.translation().dot(y.translation())) * transFactor;
}

std::pair<Eigen::Isometry3f, std::vector<bool>> optimizeSolution(const Config& cfg, const std::array<Eigen::ArrayXf, 4>& ptPairs, const std::array<Eigen::ArrayXf, 4>& tiePtPairs, const Pair<const RealCamera*>& cameras, const Isometry3f& initTrans, float huber, float recheckThreshold)
{
    Isometry3f trans = initTrans;
    // std::vector<bool> mask = recheckInliers(ptPairs, trans, cameras.first->pinHole, cameras.second->pinHole, 0, INFINITY, 0, M_PI, recheckThreshold * 16);
    // zMin should be applied here to avoid back-side points, as BA cannot move back-side points to front-side.
    //@fixme: set this to false when issue #10 is done.
    bool const checkIsFront = true;
    std::vector<bool> mask = recheckInliers(ptPairs, trans, cameras.first->pinHole, cameras.second->pinHole, recheckThreshold * 16, checkIsFront);
    auto nbInliers = countInliers(mask);
    for (unsigned i = 0; i < cfg.pairSolver.nbRefineIterations; i++)
    {
        Isometry3f newTrans = refineSolution(trans, ptPairs, mask, tiePtPairs, cfg.pairSolver.tiePtWeight, cameras.first->pinHole, cameras.second->pinHole, huber).first;
        auto newMask = recheckInliers(ptPairs, newTrans, cameras.first->pinHole, cameras.second->pinHole, recheckThreshold, checkIsFront);
        // newMask = recheckInliers(ptPairs, newTrans, cameras.first->pinHole, cameras.second->pinHole, cfg.pairSolver.zMin, cfg.pairSolver.zMax, cfg.pairSolver.minAngle, cfg.pairSolver.maxAngle, recheckThreshold);
        const float delta = solutionDiff(newTrans, trans);
        auto newNbInliers = countInliers(mask);
        if (newNbInliers < nbInliers) {
            break;
        }
        else {
            trans = newTrans;
            mask = std::move(newMask);
            nbInliers = newNbInliers;
        }
        if (std::abs(delta) < cfg.pairSolver.refineConvergThresAngle) {
            break;
        }
        if (nbInliers < cfg.pairSolver.minNbInliers) {
            break;
        }
    }
    return std::make_pair(trans, std::move(mask));
}

ImagePair createImgPair(Builder* builder, ImageHandle hFirst, ImageHandle hSecond,
	std::vector<Pair<Index>> kptsMatches, std::vector<Pair<Index>> tiePtMatches)
{
#if DEBUG_PRINT
    static fb::mutex dbgLock;
    std::lock_guard<fb::mutex> lk{dbgLock};
#endif
    const Config& cfg = builder->config();

	const uint32_t nbTiePts = cast32u(tiePtMatches.size());

    const Pair<const Image*> images {builder->getImage(hFirst), builder->getImage(hSecond)};

#ifndef NDEBUG
	// sorted in Builder::addImage
	assert(std::is_sorted(images.first->tiePtMeasurements.begin(), images.first->tiePtMeasurements.end(), MemberLess<TiePtMeasurementExt, TiePtHandle, &TiePtMeasurementExt::hTiePt>{}));
	assert(std::is_sorted(images.second->tiePtMeasurements.begin(), images.second->tiePtMeasurements.end(), MemberLess<TiePtMeasurementExt, TiePtHandle, &TiePtMeasurementExt::hTiePt>{}));
	const auto tiePtObPairs = [&]() {
		std::vector<Pair<Index>> tiePtObPairs;
		const auto& a = images.first->tiePtMeasurements;
		const auto& b = images.second->tiePtMeasurements;
		Index i = 0;
		Index j = 0;
		while (i < a.size() && j < b.size()) {
			const auto x = a[i].hTiePt;
			const auto y = b[j].hTiePt;
			if (x == y) {
				tiePtObPairs.emplace_back(i++, j++);
			}
			else if (x < y) {
				i++;
			}
			else {
				j++;
			}
		}
		return tiePtObPairs;
	}();
	assert(tiePtObPairs == tiePtMatches);
#endif
    if (kptsMatches.size() < cfg.pairSolver.minNbInliers && tiePtMatches.size() < 4) {
        return ImagePair{{hFirst, hSecond}, std::move(kptsMatches), std::move(tiePtMatches), {}};
    }

    const Pair<const RealCamera*> cameras {builder->getRealCamera(images.first->hCamera), builder->getRealCamera(images.second->hCamera)};

    const auto ptPairs = makePtPair<false>(builder, *images.first, *images.second, kptsMatches, true);
	const auto tiePtPairs = makePtPair<true>(builder, *images.first, *images.second, tiePtMatches, true);

    const int minImgDim = std::min(std::min(images.first->width, images.first->height), std::min(images.second->width, images.second->height));
    const float huber = cfg.pairSolver.refineRelativeHuberDelta * minImgDim;
    const float recheckThreshold = minImgDim * cfg.pairSolver.recheckRelativeThreshold;

    auto refineSolutions = [&](EigenAlignedVector<std::pair<Isometry3f, int64_t>> const& solutions){
        EigenAlignedVector<std::tuple<Isometry3f, std::vector<bool>, float>> candidateSolutions;
        for (const auto& [initTrans, initNbInliers] : solutions) {
            auto [trans, mask] = optimizeSolution(cfg, ptPairs, tiePtPairs, cameras, initTrans, huber, recheckThreshold);
            const float score = estimateScore(trans, gatherInliers(ptPairs, mask).second,
                int2{images.first->width, images.first->height},
                int2{images.second->width, images.second->height},
                cameras.first->pinHole, cameras.second->pinHole,
                cfg.pairSolver.recheckRelativeThreshold);
            if (score > (0.28894262f * 0.5f) * (0.5f * cfg.pairSolver.minNbInliers) * 2
                && countInliers(mask) >= cfg.pairSolver.minNbInliers) {
                candidateSolutions.emplace_back(trans, std::move(mask), score);
            }
        }
        std::sort(candidateSolutions.begin(), candidateSolutions.end(), [](const auto& a, const auto& b){
            return std::get<2>(a) > std::get<2>(b);});
        return candidateSolutions;
    };
    EigenAlignedVector<std::tuple<Isometry3f, std::vector<bool>, float>> candSolutionsF, candSolutionsH;
    if (!cfg.pairSolver.useDegensac) {
        enum class Method {kEpipolairty, kHomography};
        auto solveByRansac = [&](Method method) {
            using SolveFuncType = EigenAlignedVector<std::pair<Isometry3f, int64_t>> (*)(const Config&,
                const std::array<ArrayXf, 4>&, const std::array<ArrayXf, 4>&, int, int, const PinHoleCamera&, const PinHoleCamera&);
            static_assert(std::is_same<SolveFuncType, decltype(&solveByHomography)>::value);
            SolveFuncType solveFunc = nullptr;
            const char* methodName = nullptr;
            switch(method)
            {
            case Method::kEpipolairty: solveFunc = &solveByEpipolarity; methodName = "epipolarity"; break;
            case Method::kHomography: solveFunc = &solveByHomography; methodName = "homography"; break;
            }
            try {
                const auto solutions = solveFunc(cfg, ptPairs, tiePtPairs,
                    std::min(images.first->width, images.first->height), std::min(images.second->width, images.second->height),
                    cameras.first->pinHole, cameras.second->pinHole);
                return refineSolutions(solutions);
            }
            catch(const cudapp::Exception& e){
                printf("Warning: Failed to solve pair (%u, %u) with %s: \n\t%s:%d: %s\n",
                    static_cast<uint32_t>(hFirst), static_cast<uint32_t>(hSecond),
                    methodName, e.getFile(), e.getLine(), e.what());
            }
            return EigenAlignedVector<std::tuple<Isometry3f, std::vector<bool>, float>>{};
        };
        candSolutionsF = solveByRansac(Method::kEpipolairty);
        candSolutionsH = solveByRansac(Method::kHomography);
    }
    else {
        EigenAlignedVector<std::pair<Isometry3f, int64_t>> solutionsF, solutionsH;
        try {
            std::tie(solutionsF, solutionsH) = solveByDegensacForBoth(cfg, ptPairs, tiePtPairs,
                std::min(images.first->width, images.first->height), std::min(images.second->width, images.second->height),
                cameras.first->pinHole, cameras.second->pinHole);
        }
        catch (const cudapp::Exception& e){
            printf("Warning: Failed to solve pair (%u, %u) with %s: \n\t%s:%d: %s\n",
                static_cast<uint32_t>(hFirst), static_cast<uint32_t>(hSecond),
                "DEGENSAC", e.getFile(), e.getLine(), e.what());
        }
#if DEBUG_PRINT
        printf("before refine:\n");
        for (const auto& s : solutionsF) {
            const auto t = fromEigen(std::get<0>(s));
            printf("F: {%f, %f, %f, %f}, {%f, %f, %f}, %ld\n", t.R.w, t.R.x, t.R.y, t.R.z, t.t.x, t.t.y, t.t.z, std::get<1>(s));
        }
        for (const auto& s : solutionsH) {
            const auto t = fromEigen(std::get<0>(s));
            printf("H: {%f, %f, %f, %f}, {%f, %f, %f}, %ld\n", t.R.w, t.R.x, t.R.y, t.R.z, t.t.x, t.t.y, t.t.z, std::get<1>(s));
        }
#endif
        try {
            candSolutionsF = refineSolutions(solutionsF);
        }
        catch (const cudapp::Exception& e){
            printf("Warning: Failed to solve pair (%u, %u) with %s: \n\t%s:%d: %s\n",
                static_cast<uint32_t>(hFirst), static_cast<uint32_t>(hSecond),
                "DEGENSAC/epipolarity", e.getFile(), e.getLine(), e.what());
        }
        try {
            candSolutionsH = refineSolutions(solutionsH);
        }
        catch (const cudapp::Exception& e){
            printf("Warning: Failed to solve pair (%u, %u) with %s: \n\t%s:%d: %s\n",
                static_cast<uint32_t>(hFirst), static_cast<uint32_t>(hSecond),
                "DEGENSAC/homography", e.getFile(), e.getLine(), e.what());
        }
    }
#if DEBUG_PRINT
    printf("after refine:\n");
    for (const auto& s : candSolutionsF) {
        const auto t = fromEigen(std::get<0>(s));
        printf("F: {%f, %f, %f, %f}, {%f, %f, %f}, %ld\n", t.R.w, t.R.x, t.R.y, t.R.z, t.t.x, t.t.y, t.t.z, countInliers(std::get<1>(s)));
    }
    for (const auto& s : candSolutionsH) {
        const auto t = fromEigen(std::get<0>(s));
        printf("H: {%f, %f, %f, %f}, {%f, %f, %f}, %ld\n", t.R.w, t.R.x, t.R.y, t.R.z, t.t.x, t.t.y, t.t.z, countInliers(std::get<1>(s)));
    }
#endif
    // If there is a good match between F and H solutions, remove all other solutions.
    if (!candSolutionsF.empty() && candSolutionsH.empty()) {
        uint32_t iBest = 0;
        uint32_t jBest = 0;
        float diffBest = INFINITY;
        for (uint32_t i = 0; i < candSolutionsF.size(); i++) {
            for (uint32_t j = 0; j < candSolutionsH.size(); j++) {
                const float diff = solutionDiff(std::get<0>(candSolutionsF.at(i)), std::get<0>(candSolutionsH.at(j)));
                if (diff < diffBest) {
                    diffBest = diff;
                    i = iBest;
                    j = jBest;
                }
            }
        }
        if (diffBest < cfg.pairSolver.duplicateRotationThreshold) {
            const auto& [transF, maskF, scoreF] = candSolutionsF.at(iBest);
            const auto& [transH, maskH, scoreH] = candSolutionsH.at(jBest);
            if (scoreF > scoreH) {
                candSolutionsH.clear();
                candSolutionsF.at(0).swap(candSolutionsF.at(iBest));
                candSolutionsF.resize(1);
            }
            else {
                candSolutionsF.clear();
                candSolutionsH.at(0).swap(candSolutionsH.at(jBest));
                candSolutionsH.resize(1);
            }
        }
    }

    auto solutions = candSolutionsF;
    solutions.insert(solutions.end(), candSolutionsH.begin(), candSolutionsH.end());
    std::sort(solutions.begin(), solutions.end(), [](const auto& a, const auto& b){
        return std::get<2>(a) > std::get<2>(b);});


    for (auto iter = solutions.begin(); iter < solutions.end(); iter++) {
        const auto& [sol, mask, score] = *iter;
        if (score < std::get<2>(solutions.at(0)) * cfg.pairSolver.solutionDisambiguityScoreRatio) {
            solutions.erase(iter, solutions.end());
            break;
        }
    }

    ImagePair imgPair{{hFirst, hSecond}, std::move(kptsMatches), std::move(tiePtMatches), {}, 0, INFINITY};
    imgPair.solutions.reserve(solutions.size());
    for (const auto& [rt, oldMask, score] : solutions) {
        auto& dst = imgPair.solutions;
        Eigen::MatrixX3f const pts3d = triangulate(cameras.first->pinHole, cameras.second->pinHole, rt, ptPairs);
        const auto mask = recheckInliers(ptPairs, rt, cameras.first->pinHole, cameras.second->pinHole, recheckThreshold, true, &pts3d);
        //@fixme: recompute score?
        auto inliers = mask2indices(mask);
        // If the baseline for inlier pairs are too short, we give up the all solutions, so a worse solution won't be used later.
        auto const[medianZ, medianAngleCPC] = getMedianZAndAngleCPC(rt, pts3d, inliers, reinterpret_cast<uint64_t const&>(imgPair.images)); static_assert(sizeof(imgPair.images) == 8);
        if (!inRange(medianZ, cfg.pairSolver.zMin, cfg.pairSolver.zMax) && !inRange(medianAngleCPC, cfg.pairSolver.minAngle, cfg.pairSolver.maxAngle)) {
#if DEBUG_PRINT
            printf("[debug] reject all solutions due to small baseline in one solution\n");
#endif
            solutions.clear();
            dst.clear();
            break;
        }
        imgPair.maxMedianDepth = std::max(imgPair.maxMedianDepth, medianZ);
        imgPair.minMedianAngle = std::min(imgPair.minMedianAngle, medianAngleCPC);

		// If we have more tie points, we relax the threshold.
		const float thresholdRelaxing = clamp(1.f - std::tanh(nbTiePts / 8.f) / std::tanh(8.f / 8.f), 0.f, 1.f);
        const float scoreThreshold = std::get<2>(solutions.at(0)) * cfg.pairSolver.solutionDisambiguityScoreRatio * thresholdRelaxing;
        const bool isDuplicate = std::any_of(dst.begin(), dst.end(), [&rt, &cfg](const ImagePair::Solution& s){
            return solutionDiff(rt, toEigen(s.transform)) < cfg.pairSolver.duplicateRotationThreshold;
        });
        if (!isDuplicate && score >= scoreThreshold && inliers.size() >= cfg.pairSolver.minNbInliers * thresholdRelaxing)
        {
            dst.emplace_back(ImagePair::Solution{fromEigen(rt), std::move(inliers), score});
        }
    }

#if DEBUG_PRINT
    std::stringstream ss;
    ss << static_cast<uint32_t>(hFirst) << "-" << static_cast<uint32_t>(hSecond) << "," << imgPair.kptsMatches.size() << ": \n";
    for (const auto& s : imgPair.solutions) {
        ss << "\t{" << s.transform.R.w << "," << s.transform.R.x << "," << s.transform.R.y << "," << s.transform.R.z << "}, " << "{" << s.transform.t.x << "," << s.transform.t.y << "," << s.transform.t.z << "}, "
           << s.inliers.size() << "\n";
    }
    printf("%s\n", ss.str().c_str());
#endif
    return imgPair;
}

// @fixme: add test
float estimateScore(const Isometry3f& rt, const Eigen::MatrixX4f& inlierPtPairs, const int2 firstImgSize, const int2 secondImgSize,
const PinHoleCamera& firstCam, const PinHoleCamera& secondCam, float relativeThreshold) {
    if (inlierPtPairs.rows() < 2) {
        return 0;
    }
#if 1
    std::array<Eigen::MatrixX3f, 2> pts3d;
    pts3d[0] = triangulate(firstCam, secondCam, rt, inlierPtPairs);
    pts3d[1] = (rt * pts3d[0].transpose()).transpose();

    Eigen::MatrixX4f point_pairs_proj(inlierPtPairs.rows(), inlierPtPairs.cols());
    point_pairs_proj << (toKMat<float>(firstCam) * pts3d[0].transpose()).colwise().hnormalized().transpose(),
            (toKMat<float>(secondCam) * pts3d[1].transpose()).colwise().hnormalized().transpose();
    Eigen::VectorXf sqr_error[2] = {
            (inlierPtPairs.template leftCols<2>() - point_pairs_proj.template leftCols<2>()).rowwise().squaredNorm().eval(),
            (inlierPtPairs.template rightCols<2>() - point_pairs_proj.template rightCols<2>()).rowwise().squaredNorm().eval()
    };
    const float errorThreshold = std::min(firstImgSize.x, secondImgSize.x) * relativeThreshold;
    auto error2score = [&](const auto& err) {return (0.5f - 0.5f * (err.array() * 0.5f - (errorThreshold / 2) * 0.5f).tanh()).eval();};
    const float errScore = error2score(sqr_error[0]).sum() + error2score(sqr_error[1]).sum();
#else
    unused(rt, firstCam, secondCam, relativeThreshold);
    float const errScore = inlierPtPairs.rows();
#endif
    //consider point 2d span factor
    Vector2f span = Vector2f::Zero();
    for(int i = 0; i < 2; i++)
    {
        //@todo: consider center and direction of span to make it more accurate
        const Vector2f sqrt_singular_values = computeCovariance(inlierPtPairs.template middleCols<2>(i * 2).eval()).jacobiSvd().singularValues().cwiseSqrt();
//        const float span_rep = sqrt_singular_values.prod() / sqrt_singular_values.mean();
        const float span_rep = std::sqrt(sqrt_singular_values.prod());

//        span[i] = span_rep  / img_pair->images()[i]->get_camera()->f.mean();
        const auto& img_size = (i == 0 ? firstImgSize : secondImgSize);
        span[i] = span_rep  / std::min(img_size.x, img_size.y);
    }

//    const float span_factor = span.prod() / span.mean();
    const float span_factor = std::sqrt(span.prod());
    const float score = errScore * span_factor;

    return score;
}

// @fixme: add unit test
ImagePair ImagePair::inverse() const {
    ImagePair ret{
        {images.second, images.first},
        {},
        {},
		{}
    };
    ret.kptsMatches.reserve(kptsMatches.size());
    std::transform(kptsMatches.begin(), kptsMatches.end(), std::back_inserter(ret.kptsMatches),
        [](const Pair<Index>& m){ return std::make_pair(m.second, m.first); });
	ret.tiePtMatches.reserve(tiePtMatches.size());
	std::transform(tiePtMatches.begin(), tiePtMatches.end(), std::back_inserter(ret.tiePtMatches),
        [](const Pair<Index>& m){ return std::make_pair(m.second, m.first); });
    ret.solutions.reserve(solutions.size());
    std::transform(solutions.begin(), solutions.end(), std::back_inserter(ret.solutions), [](const Solution& s){
        return Solution{s.transform.inverse(), s.inliers, s.score};
    });
    return ret;
}
} // namespace rsfm
