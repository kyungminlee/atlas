#include <cstddef>
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include "util.hpp"
#include "multi_array.hpp"

    // using namespace atlas;
    // fmt::print("Hi\n");

    // // std::array<int, 2> lower{2,3};
    // // std::array<int, 2> upper{5,8};
    // // atlas::simple_multi_indexer<2, size_t, int> indexer(lower, upper);

    // // multi_array<double, 2, size_t, int> arr(indexer);
    // multi_array<double, 2, size_t, int> arr({2,3}, {5,8});

    // int index = 0;
    // for (auto & v: arr) {
    //     v = 10.001*index;
    //     ++index;
    // }
    // for (auto i: arr.indexer()) {
    //     std::cout << i.index << ", " << i.multi_index << " = " << arr[i.index] << std::endl;
    // }
    


    // SECTION("iterator") {
    //     //          2   3   4   5   6
    //     //      1 | 0   3   6   9   12
    //     //      2 | 1  *4  *7  *10  13
    //     //      3 | 2  *5  *8  *11  14

    //     // TODO: move to test-array
    //     SECTION("access by multi_index") {
    //         basic_multi_array<double, decltype(indexer)> arr(indexer);
    //         for (auto item: indexer) {
    //             arr(item.multi_index) = item.index + 100 * item.compact_index;
    //         }
    //         std::vector<double> test_arr(arr.data(), arr.data() + indexer.size());
    //         std::vector<double> true_arr = {0, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414};
    //         REQUIRE((test_arr == true_arr));
    //     }
    //     std::cout << "SLICE\n";
    //     auto slindexer = indexer.slice({2,3}, {4,6});
    //     {
    //         basic_multi_array<double, decltype(slindexer)> arr(slindexer);
    //         for (auto item: slindexer) {
    //             arr(item.multi_index) = item.index + 100 * item.compact_index;
    //         }
    //         for (int i = 0 ; i < slindexer.size() ; ++i) {
    //             std::cout << arr.data()[i] << std::endl;
    //         }
    //     }
    // }



    // {
    //     //          2   3   4   5   6
    //     //      1 | 0   3   6   9   12
    //     //      2 | 1  *4  *7  *10  13
    //     //      3 | 2  *5  *8  *11  14
    //     simple_multi_indexer<2, size_t, int> indexer({1,2}, {4,7});
    //     {
    //         basic_multi_array<double, decltype(indexer)> arr(indexer);
    //         std::cout << "START\n";
    //         for (auto item: indexer) {
    //             arr(item.multi_index) = item.index + 100 * item.compact_index;
    //         }
    //         for (int i = 0 ; i < indexer.size() ; ++i) {
    //             std::cout << arr.data()[i] << std::endl;
    //         }
    //     }
    //     std::cout << "SLICE\n";
    //     auto slindexer = indexer.slice({2,3}, {4,6});
    //     {
    //         basic_multi_array<double, decltype(slindexer)> arr(slindexer);
    //         for (auto item: slindexer) {
    //             arr(item.multi_index) = item.index + 100 * item.compact_index;
    //         }
    //         for (int i = 0 ; i < slindexer.size() ; ++i) {
    //             std::cout << arr.data()[i] << std::endl;
    //         }
    //     }
    // }
