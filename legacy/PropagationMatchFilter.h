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
#include <vector>
#include <cstddef>
#include <eigen3/Eigen/Dense>
#include "../Types.hpp"
#include <cpp_utils.h>

namespace rsfm::pmf {
class PropagationMatchFilter {
public:
    static constexpr int halo = 1;
    static constexpr float threshold_relative = 0.15f;
    static constexpr float ransac_confidence = 0.25f;
    static constexpr size_t ransac_max_iterations = 32;
    using Affine2f = Eigen::Transform<float, 2, Eigen::AffineCompact>;
    using PtPair = std::pair<Vec2f, Vec2f>;
public:
    PropagationMatchFilter() = default;
    PropagationMatchFilter(Vec2<int> imgSize0, const std::vector<PtPair>& matches) {
        init(imgSize0, matches);
    }

    std::vector<bool> getInlierMask(int minVotes = 2) const;

private:
    void init(Vec2<int> imgSize0, const std::vector<PtPair>& matches);

    struct Params {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Affine2f trans;
        uint32_t numInliers;
    };

    const std::vector<uint32_t> &cell(int i, int j) const { return mCells.at(cast32u(i * mCols + j)); }

    std::vector<uint32_t> &cell(int i, int j) { return mCells.at(cast32u(i * mCols + j)); }

    const Params &param(int i, int j) const { return mParams.at(cast32u(i * mCols + j)); }

    Params &param(int i, int j) { return mParams.at(cast32u(i * mCols + j)); }

    enum Direction {
        Forward,
        Backward
    };

    void processCell(int y, int x, Direction direction, bool vote = false);

    void propagate(Direction direction, bool vote = false);

    void run();

private:
    int mCellWidth = 0;
    std::vector<std::vector<uint32_t>> mCells;
    int mCols = 0;
    int mRows = 0;
    const std::vector<PtPair> *mMatches = nullptr;
    std::vector<Params, Eigen::aligned_allocator<Params>> mParams;
    std::vector<int> mVotes;
private: //buffers used to reduce new/delete
    std::vector<uint32_t> input;
};
} // namespace rsfm/pmf
