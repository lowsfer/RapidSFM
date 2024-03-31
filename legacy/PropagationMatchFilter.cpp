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

//
// Created by yao on 17/06/18.
//
#ifdef NDEBUG
#pragma GCC optimize("-ffast-math", "-fno-finite-math-only")
#endif

#include "PropagationMatchFilter.h"
#include "geometry.hpp"
#include "../ransac.hpp"
#include <cpp_utils.h>

namespace rsfm::pmf
{

void PropagationMatchFilter::init(Vec2<int> imgSize0, const std::vector<PtPair>& matches)
{
    if(imgSize0.x >= imgSize0.y) {
        mCols = 16;
        mCellWidth = divUp(imgSize0.x, mCols);
        mRows = divUp(imgSize0.y, mCellWidth);
    }else{
        mRows = 16;
        mCellWidth = divUp(imgSize0.y, mRows);
        mCols = divUp(imgSize0.x, mCellWidth);
    }
    mCells.resize(cast32u(mCols * mRows));
    for(auto& cell : mCells)
        cell.clear();
    mMatches = &matches;
    const float cellWidthInv = 1.f / mCellWidth;
    for(unsigned i = 0; i < matches.size(); i++)
    {
        const auto& m = matches[i];
        const auto& pt = m.first;
        cell(int(std::floor(pt.y * cellWidthInv)), int(std::floor(pt.x * cellWidthInv))).push_back(i);
    }
    mParams.resize(cast32u(mCols * mRows));
    for(auto& param : mParams)
        param = {Affine2f::Identity(), 0};
    mVotes.resize(mMatches->size());
    std::fill(mVotes.begin(), mVotes.end(), 0);

    run();
}

void PropagationMatchFilter::processCell(int y, int x, Direction direction, bool vote)
{
//    std::cout << "Processing cell (" << y << ", " << x << ")" << std::endl;
    input.clear();
    for (int i = int(y) - halo; i <= int(y) + halo; i++)
        for (int j = int(x) - halo; j <= int(x) + halo; j++)
            if(i >= 0 && i < int(mRows) && j >= 0 && j < int(mCols))
                input.insert(input.end(), cell(i, j).begin(), cell(i, j).end());
    if(input.size() < 8) {
        param(y,x) = {Affine2f::Identity(), 0};
        return;
    }
    std::array<Eigen::ArrayXf, 4> point_pairs;
    Eigen::ArrayXf& ax = point_pairs[0]; ax.resize(cast32i(input.size()));
    Eigen::ArrayXf& ay = point_pairs[1]; ay.resize(cast32i(input.size()));
    Eigen::ArrayXf& bx = point_pairs[2]; bx.resize(cast32i(input.size()));
    Eigen::ArrayXf& by = point_pairs[3]; by.resize(cast32i(input.size()));
    for(int i = 0; i < int(input.size()); i++)
    {
        const auto& match = (*mMatches)[input[uint32_t(i)]];
        const Vec2f& a = match.first;
        ax[i] = a.x;
        ay[i] = a.y;
        const Vec2f& b = match.second;
        bx[i] = b.x;
        by[i] = b.y;
    }

    auto indices2pts = [&](const std::array<uint32_t, 3>& indices) -> Eigen::Matrix<float, 3, 4>{
        Eigen::Matrix<float, 3, 4> pts;
        for(size_t i = 0; i < indices.size(); i++)
        {
            const uint32_t idx = indices[i];
            pts.row(i) <<  ax[idx], ay[idx], bx[idx], by[idx];
        }
        return pts;
    };

    auto ransac_test = [&](const std::array<uint32_t, 3>& indices) -> uint32_t {
        Eigen::Matrix<float, 3, 4> pts = indices2pts(indices);
#if 1
        const Affine2f trans = findAffine2(pts);
#else
        const Affine2f trans = findSim2(pts.template topRows<2>());
#endif
        const float sqrScale = trans.matrix().template leftCols<2>().determinant();
        // we do not allow negative determinant, as that means flipping
        if(sqrScale < square(0.25f) || sqrScale > square(4.f) || !std::isfinite(sqrScale))
            return 0u;

        return (uint32_t)checkAffine2Transformation(trans, point_pairs, mCellWidth * threshold_relative).count();
    };

    std::array<uint32_t, 3> best_indices = ransac<decltype(ransac_test), 3, uint32_t>(ransac_test, input.size(), ransac_confidence, ransac_max_iterations, y * mCols + x);
    const Affine2f trans_ransac = findAffine2(indices2pts(best_indices));
    Affine2f trans_neighbour0 = Affine2f::Identity();
    Affine2f trans_neighbour1 = Affine2f::Identity();
    if(direction == Direction::Forward){
        if(x > 0)
            trans_neighbour0 = param(y, x-1).trans;
        if(y > 0)
            trans_neighbour1 = param(y-1, x).trans;
    }
    else
    {
        if(x < mCols - 1)
            trans_neighbour0 = param(y, x + 1).trans;
        if(y < mRows - 1)
            trans_neighbour1 = param(y + 1, x).trans;
    }
    Eigen::Array<bool, Eigen::Dynamic, 1> mask;
    uint32_t numInliers = 0;
    for(auto trans : {trans_ransac, trans_neighbour0, trans_neighbour1})
    {
        mask = checkAffine2Transformation(trans, point_pairs, mCellWidth * threshold_relative);
        numInliers = uint32_t(mask.count());
        //refine trans
        {
            Eigen::Matrix<float, -1, 4> pts(numInliers, 4);
            uint32_t p = 0;
            for(size_t i = 0; i < input.size(); i++)
            {
                if(mask[i])
                    pts.row(p++) <<  ax[i], ay[i], bx[i], by[i];
            }
            assert(p == numInliers);
            trans = findAffine2(pts);
            mask = checkAffine2Transformation(trans, point_pairs, mCellWidth * threshold_relative);
            numInliers = uint32_t(mask.count());
        }
        if(numInliers > param(y,x).numInliers)
            param(y, x) = {trans, numInliers};
    }
    if(vote && int(numInliers) > 4){
        for(unsigned i = 0; i < input.size(); i++){
            if(mask[i])
                mVotes[input[i]]++;
        }
    }
}

void PropagationMatchFilter::propagate(Direction direction, bool vote)
{
    if(direction == Direction::Forward) {
        for (int i = 0; i < mRows; i++) {
            for (int j = 0; j < mCols; j++) {
                processCell(i, j, direction, vote);
            }
        }
    }else{
        for (int i = mRows-1; i >= 0; i--) {
            for (int j = mCols-1; j >= 0; j--) {
                processCell(i, j, direction, vote);
            }
        }
    }
}

void PropagationMatchFilter::run()
{
    propagate(Direction::Forward, false);
    propagate(Direction::Backward, true);
}

std::vector<bool> PropagationMatchFilter::getInlierMask(int minVotes) const
{
    std::vector<bool> mask(mMatches->size());
    for(unsigned i = 0; i < mMatches->size(); i++)
        mask[i] = mVotes[i] > minVotes;
    return mask;
}
} // namespace rsfm::pmf
