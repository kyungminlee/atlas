#include <concepts>
#include <iterator>
#include <iostream>
#include <typeinfo>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include "block_range.hpp"
#include "multi_indexer.hpp"
#include "util.hpp"


template <typename Iter>
void check_iterator_concepts()
{
  using namespace std;  
#define SHOW(x) (std::cout << (#x) << "<" << typeid(Iter).name() << "> : " << x<Iter> << std::endl)
  SHOW(indirectly_readable);
  SHOW(movable);
  SHOW(weakly_incrementable);
  SHOW(input_iterator);
  // SHOW(output_iterator);
  SHOW(forward_iterator);
  SHOW(bidirectional_iterator);
  SHOW(random_access_iterator);
  SHOW(contiguous_iterator);
  SHOW(totally_ordered);
  SHOW(equality_comparable);
}

void test()
{
  using namespace atlas;


  block_range<int> r(10, 16);
  std::cout << "R: " << r << std::endl;
  std::cout << "B: " << r.begin() << std::endl;
  std::cout << r.end() << std::endl;
  std::cout << r.size() << std::endl;
  std::cout << r.empty() << std::endl;

  for (auto i: r) {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  check_iterator_concepts<decltype(r.begin())>();

  return;

  simple_multi_indexer<2> large_indexer({10, 10});
  std::vector<decltype(large_indexer)::slice_type> slices;

  for (size_t i = 0 ; i < 5 ; ++i) {
    for (size_t j = 0 ; j < 5 ; ++j) {
      auto sl = large_indexer.slice({2*i, 2*j}, {2*(i+1), 2*(j+1)});
      slices.push_back(std::move(sl));
    }
  }

  {
    std::vector<size_t> indices1, indices2;
    for (auto idx : large_indexer) {
      indices1.push_back(idx.index);
    }
    for (auto const & sl: slices) {
      fmt::print("slice: {} {} {} {}\n", sl.lower(0), sl.upper(0), sl.lower(1), sl.upper(1));
      for (auto idx: sl) {
        indices2.push_back(idx.index);
      }
    }
    fmt::print("indices1: {}\n", indices1);
    fmt::print("indices2: {}\n", indices2);
  }

  {
    std::vector<std::array<size_t, 2>> indices1, indices2;
    for (auto idx : large_indexer) {
      indices1.push_back(idx.multi_index);
    }
    for (auto const & sl: slices) {
      fmt::print("slice: {} {} {} {}\n", sl.lower(0), sl.upper(0), sl.lower(1), sl.upper(1));
      for (auto idx: sl) {
        fmt::print("Adding {} {}\n", idx.multi_index, idx.index);
        indices2.push_back(idx.multi_index);
      }
    }
    fmt::print("indices1: {}\n", indices1);
    fmt::print("indices2: {}\n", indices2);
    std::sort(indices1.begin(), indices1.end());
    std::sort(indices2.begin(), indices2.end()); 
    fmt::print("indices1: {}\n", indices1);
    fmt::print("indices2: {}\n", indices2);
    fmt::print("equal? {}\n", indices1 == indices2);
  }
}



int main()
{
  std::cout << std::boolalpha;
  test();
  return 0;

  using namespace atlas;
  using namespace std;

  simple_multi_indexer<3> midx({2,3,4}, {6,7,8});

  std::cout << "semiregular: " << std::semiregular<simple_multi_indexer<3>> << std::endl;

  auto iter = midx.begin();
  using Iter = decltype(iter);
  check_iterator_concepts<Iter>();

  std::cout << "s: " << std::sentinel_for<Iter, Iter> << std::endl;
  std::cout << "ss: " << std::sized_sentinel_for<Iter, Iter> << std::endl;
  // std::cout << "po: " << __detail::__partially_ordered_with<Iter, Iter> << std::endl;
  std::cout << "totally_ordered<void*>: " << std::totally_ordered<void*> << std::endl;
  return 0;
  
  // using T = typename std::iter_difference_t<Iter>;
  // std::cout << typeid(T).name() << std::endl;
  // std::cout << "w: " << std::boolalpha << std::weakly_incrementable<Iter> << std::endl;
  // std::cout << "io: " << std::boolalpha << std::input_or_output_iterator<Iter> << std::endl;
  // std::cout << "i: " << std::boolalpha << std::input_iterator<Iter> << std::endl;
  // std::cout << "f: " << std::boolalpha << std::forward_iterator<Iter> << std::endl;
  // std::cout << "r: " << std::boolalpha << std::random_access_iterator<Iter> << std::endl;

  std::cout << "size : " << midx.size() << std::endl;
  for (auto item: midx) {
    std::cout << item.index << ", " << item.compact_index << ", " << item.multi_index << std::endl;
  }
  std::cout << "---" << std::endl;
  multi_indexer_slice<simple_multi_indexer<3>> sl(midx, {3,4,5}, {5,6,7});
  for (auto item: sl) {
    std::cout << item.index << ", " << item.compact_index << ", " << item.multi_index << std::endl;
  }
  std::cout << "---" << std::endl;
  auto sl2 = midx.slice({4,4,5}, {5,6,7});
  for (auto item: sl2) {
    std::cout << item.index << ", " << item.compact_index << ", " << item.multi_index << std::endl;
  }
  std::cout << "---" << std::endl;
  auto sl3 = sl2.slice({4,4,6}, {5,6,7});
  for (auto item: sl3) {
    std::cout << item.index << ", " << item.compact_index << ", " << item.multi_index << std::endl;
  }
}