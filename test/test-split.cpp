#include <cstdint>
#include <limits>
#include <vector>
#include <utility>
#include <type_traits>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

// #include <catch2/catch_test_macros.hpp>
// #include <catch2/benchmark/catch_benchmark.hpp>
#include "util.hpp"

#include "multi_indexer.hpp"
#include "multi_array.hpp"
#include "gpu/cuda_mock.hpp"
#include "gpu/cuda_multi_array.hpp"

int main()
{
    using namespace atlas;
    int nx = 100, ny = 100, nz = 100;
    simple_multi_indexer<3, size_t, int> global_indexer({0,0,0}, {100, 100, 100});
    std::vector<decltype(global_indexer.slice({0,0,0}, {1,1,1}))> block_indexer_list;
    int bx = 50, by = 20, bz = 10;
    for (int ix = 0 ; ix < nx ; ix += bx) {
        for (int iy = 0 ; iy < ny ; iy += by) {
            for (int iz = 0 ; iz < nz ; iz += bz) {
                block_indexer_list.push_back(global_indexer.slice({ix, iy, iz}, {ix+bx, iy+by, iz+bz}));
            }
        }
    }

    std::vector<basic_multi_array<double, typename decltype(block_indexer_list)::value_type>> array_list;
    array_list.reserve(block_indexer_list.size());
    for (auto const & b : block_indexer_list) {
        fmt::print("{} -- {}\n", b.lower(), b.upper());
        array_list.emplace_back(b);
    }



}