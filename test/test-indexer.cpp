#include <cstdint>

#include <limits>
#include <utility>
#include <stdexcept>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "multi_indexer.hpp"
#include "multi_array.hpp"
#include "util.hpp"

template <typename T, typename U>
constexpr bool is_equiv_v = std::is_same<typename std::decay<T>::type, U>::value;


TEST_CASE("multi_indexer", "[multi_indexer]") {
    using namespace atlas;

    SECTION("comparison") {
        simple_multi_indexer<3, size_t, int> indexer1({2,3,4});
        simple_multi_indexer<3, size_t, int> indexer2({0,0,0}, {2,3,4});
        simple_multi_indexer<3, size_t, int> indexer3({1,0,0}, {2,3,4});
        REQUIRE(indexer1 == indexer1);
        REQUIRE(indexer1 == indexer2);
        REQUIRE(indexer1 != indexer3);
        REQUIRE(indexer3 == indexer3);
    }

    SECTION("simple getters") {
        std::array<int, 3> lower{1,0,0}, upper{2,3,4};
        std::array<size_t, 4> stride{1,1,3,12};
        simple_multi_indexer<3, size_t, int> indexer(lower, upper);
        REQUIRE(indexer.parent() == indexer);
        REQUIRE(indexer.size() == 1*3*4);
        for (size_t d = 0 ; d < 3 ; ++d) {
            REQUIRE(indexer.lower(d) == lower[d]);
            REQUIRE(indexer.upper(d) == upper[d]);
            REQUIRE(indexer.stride(d) == stride[d]);
            REQUIRE(indexer.upper_stride(d) == stride[d+1]);
        }
        std::vector<std::array<int, 3>> multi_indices = {
            {1,0,0}, {1,1,0}, {1,2,0},
            {1,0,1}, {1,1,1}, {1,2,1},
            {1,0,2}, {1,1,2}, {1,2,2},
            {1,0,3}, {1,1,3}, {1,2,3},
        };
        for (size_t i = 0 ; i < multi_indices.size() ; ++i) {
            std::array<int, 3> const & midx = multi_indices[i];
            REQUIRE(indexer.ravel(midx) == i);
            REQUIRE(indexer.unravel(i) == midx);
        }
        REQUIRE_THROWS_AS(indexer.unravel(-1), std::out_of_range);
        REQUIRE_THROWS_AS((indexer.ravel({0,0,0})), std::out_of_range);
        REQUIRE_THROWS_AS((indexer.ravel({2,3,4})), std::out_of_range);
    }

    SECTION("constructor-exceptions") {
        int const MAX = std::numeric_limits<int>::max();
        simple_multi_indexer<2, size_t, int> indexer({MAX, MAX});

        static_assert(is_equiv_v<decltype(indexer.lower()), std::array<int, 2>>);
        static_assert(is_equiv_v<decltype(indexer.upper()), std::array<int, 2>>);
        static_assert(is_equiv_v<decltype(indexer.size()), size_t>);
        static_assert(is_equiv_v<decltype(indexer.shape()), std::array<size_t, 2>>);
        static_assert(is_equiv_v<decltype(indexer.lower(0)), int>);
        static_assert(is_equiv_v<decltype(indexer.upper(0)), int>);
        static_assert(is_equiv_v<decltype(indexer.size(0)), size_t>);
        static_assert(is_equiv_v<decltype(indexer.ravel_compact({0,0})), size_t>);
        static_assert(is_equiv_v<decltype(indexer.unravel_compact(0)), std::array<int, 2>>);
        static_assert(is_equiv_v<decltype(indexer.ravel({0,0})), size_t>);
        static_assert(is_equiv_v<decltype(indexer.unravel(0)), std::array<int, 2>>);

        REQUIRE(indexer.size() == static_cast<size_t>(MAX) * MAX);
        REQUIRE_THROWS_AS((simple_multi_indexer<2, int, int>({MAX, MAX})), std::overflow_error);
        REQUIRE_THROWS_AS((simple_multi_indexer<3, size_t>({2,3})), std::out_of_range);
        REQUIRE_THROWS_AS((simple_multi_indexer<3, size_t, int>({2,-3,4})), std::out_of_range);
        REQUIRE_THROWS_AS((simple_multi_indexer<2, char, int>({1024, 1024})), std::overflow_error);
    }

    SECTION("indexing") {
        simple_multi_indexer<3, size_t, int> indexer({1,0,0}, {2,3,4});
        std::vector<size_t> compact_indices; compact_indices.reserve(indexer.size());
        std::vector<size_t> indices; indices.reserve(indexer.size());
        std::vector<std::array<int, 3>> multi_indices; multi_indices.reserve(indexer.size());
        for (auto item: indexer) {
            compact_indices.push_back(item.compact_index);
            indices.push_back(item.index);
            multi_indices.push_back(item.multi_index);
        }
        REQUIRE(compact_indices.size() == indexer.size());
        REQUIRE(compact_indices == std::vector<size_t>{0,1,2,3,4,5,6,7,8,9,10,11});
        REQUIRE(indices == std::vector<size_t>{0,1,2,3,4,5,6,7,8,9,10,11});
    }

    SECTION("offset_multi_indexer") {
        simple_multi_indexer<3, size_t, int> indexer({1,0,0}, {2,3,4});
        offset_multi_indexer<simple_multi_indexer<3, size_t, int>> offset_indexer(indexer, 3);
        std::vector<size_t> compact_indices; compact_indices.reserve(indexer.size());
        std::vector<size_t> indices; indices.reserve(indexer.size());
        std::vector<std::array<int, 3>> multi_indices; multi_indices.reserve(indexer.size());
        for (auto item: offset_indexer) {
            compact_indices.push_back(item.compact_index);
            indices.push_back(item.index);
            multi_indices.push_back(item.multi_index);
        }
        REQUIRE(compact_indices.size() == indexer.size());
        REQUIRE(compact_indices == std::vector<size_t>{0,1,2,3,4,5,6,7,8,9,10,11});
        REQUIRE(indices == std::vector<size_t>{3,4,5,6,7,8,9,10,11,12,13,14});
    }

    SECTION("slice") {
        //          2   3   4   5   6
        //      1 | 0   3   6   9   12
        //      2 | 1  *4  *7  *10  13
        //      3 | 2  *5  *8  *11  14
        simple_multi_indexer<2, size_t, int> indexer({1,2}, {4,7});
        auto slice = indexer.slice({2,3}, {4,6});
        REQUIRE(slice.size() == 2*3);
        REQUIRE((slice.size(0) == 2 && slice.size(1) == 3));
        REQUIRE((slice.shape(0) == 2 && slice.shape(1) == 3));
        REQUIRE((slice.lower(0) == 2 && slice.lower(1) == 3));
        REQUIRE((slice.upper(0) == 4 && slice.upper(1) == 6));
        REQUIRE((slice.ravel({2,3}) == 4 && slice.unravel(4) == std::array<int, 2>({2,3})));
        REQUIRE((slice.ravel({3,4}) == 8 && slice.unravel(8) == std::array<int, 2>({3,4})));
 
        // REQUIRE_THROWS_AS(slice.ravel({}))
        REQUIRE_THROWS_AS(slice.ravel({1,2}), std::out_of_range);
        REQUIRE_THROWS_AS(slice.unravel(13), std::out_of_range);

        std::vector<size_t> compact_indices; compact_indices.reserve(slice.size());
        std::vector<size_t> indices; indices.reserve(slice.size());
        std::vector<std::array<int, 2>> multi_indices; multi_indices.reserve(slice.size());

        for (auto item : slice) {
            compact_indices.push_back(item.compact_index);
            indices.push_back(item.index);
            multi_indices.push_back(item.multi_index);
            REQUIRE(slice.ravel(item.multi_index) == item.index);
            REQUIRE(slice.unravel(item.index) == item.multi_index);
        }
        REQUIRE(compact_indices == std::vector<size_t>({0,1,2,3,4,5}));
        REQUIRE(indices == std::vector<size_t>({4,5,7,8,10,11}));
        REQUIRE(multi_indices == std::vector<std::array<int, 2>>({ {2,3},{3,3},{2,4},{3,4},{2,5},{3,5} }));
    }

    SECTION("pad slice") {
        simple_multi_indexer<3, size_t, unsigned int> idx({2,3,4}, {10, 11, 12});
        auto sl = idx.slice({5,7,8}, {9,10,11});
        int lx = 5, ly = 7, lz = 8;
        int ux = 9, uy = 10, uz = 11;
        for (int i = 0 ; i < 10 ; ++i) {
            REQUIRE((lx >= 0 && ly >= 0 && lz >= 0 && ux >= 0 && uy >= 0 && uz >= 0));
            {
                unsigned int lx_u = lx, ly_u = ly, lz_u = lz, ux_u = ux, uy_u = uy, uz_u = uz;
                auto pl = sl.pad(i);
                REQUIRE((pl.lower() == std::array<unsigned int, 3>{lx_u, ly_u, lz_u}));
                REQUIRE((pl.upper() == std::array<unsigned int, 3>{ux_u, uy_u, uz_u}));
            }
            std::tie(lx, ly, lz) = std::tie(std::max(lx-1, 2), std::max(ly-1, 3), std::max(lz-1, 4));
            std::tie(ux, uy, uz) = std::tie(std::min(ux+1,10), std::min(uy+1,11), std::min(uz+1,12));
        }

        SECTION("Pad by a very large number") {
            auto pl = sl.pad(std::numeric_limits<unsigned int>::max());
            REQUIRE(pl.lower() == idx.lower());
            REQUIRE(pl.upper() == idx.upper());
        }
    }
}

TEST_CASE("offset_multi_indexer", "[offset_multi_indexer]") {
}