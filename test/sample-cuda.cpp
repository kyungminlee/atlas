#include <cstdint>
#include <limits>
#include <vector>

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "multi_indexer.hpp"
#include "multi_array.hpp"
#include "gpu/cuda_mock.hpp"
#include "gpu/cuda_multi_array.hpp"

void test_function(double const * p) {
    fmt::print("received {}\n", reinterpret_cast<void const *>(p));
}

TEST_CASE("cuda", "[cuda]") {
    using namespace atlas;
    simple_multi_indexer<2, size_t, int> indexer({1,2}, {4,7});
    std::vector<multi_array_cuda<double, decltype(indexer)>> vec;

    for (int i = 0 ; i < 5 ; ++i) {
        vec.push_back(make_multi_array_cuda<double>(i, indexer));
    }
    for (int i = 0 ; i < 5 ; ++i) {
        auto v = vec[i].data();
        fmt::print("type of object: {}\n", typeid(v).name());
        test_function(v);
    }
}
