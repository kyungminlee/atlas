#pragma once

///@file Util.h
///@brief Provides utility functions.
///@author Kyungmin Lee

/*************************************************************************
*
* SEC CONFIDENTIAL
* __________________
*
*  [2008-] Samsung Electronics Co., Ltd.
*  All Rights Reserved.
*
* NOTICE:  All information contained herein is, and remains
* the property of Samsung Electronics Co., Ltd. and its suppliers,
* if any.  The intellectual and technical concepts contained
* herein are proprietary to Samsung Electronics Co., Ltd. 
* and its suppliers and may be covered by South Korea, U.S., Japan and 
* other Foreign Patents, patents in process, and are protected by 
* trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Samsung Electronics Co., Ltd.
**************************************************************************/

#include <iostream>
#include <utility>
#include <tuple>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>


namespace std {
  template <typename Ch, typename Tr, typename T1, typename T2>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr>& os, pair<T1, T2> const & arr)
  {
    return os << '(' << arr.first << ", " << arr.second << ')';
  }

  namespace aux {
    template<size_t...> struct seq {};

    template<size_t N, size_t... Is>
    struct gen_seq : gen_seq<N-1, N-1, Is...> {};

    template<size_t... Is>
    struct gen_seq<0, Is...> : seq<Is...> {};

    template<typename Ch, typename Tr, typename Tuple, size_t... Is>
    void print_tuple(basic_ostream<Ch,Tr> & os, Tuple const & t, seq<Is...>){
      using swallow = int[];
      (void)swallow{0, (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
    }
  } // namespace aux

  template<typename Ch, typename Tr, typename ... Args>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, tuple<Args...> const & t)
  {
    os << '(';
    aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
    return os << ')';
  }

  template <typename Ch, typename Tr, typename T>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, vector<T> const & arr)
  {
    os << '[';
    char const * sep = "";
    for (auto const & item: arr) {
      os << sep << item;
      sep = ", ";
    }
    os << ']';
    return os;
  }

  template <typename Ch, typename Tr, typename T, size_t D>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, array<T, D> const & arr) {
    os << '[';
    char const * sep = "";
    for (auto const & item: arr) {
      os << sep << item;
      sep = ", ";
    }
    os << ']';
    return os;
  }

  template <typename Ch, typename Tr, typename T>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, set<T> const & arr) {
    os << '[';
    char const * sep = "";
    for (auto const & item: arr) {
      os << sep << item;
      sep = ", ";
    }
    os << ']';
    return os;
  }

  template <typename Ch, typename Tr, typename T>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, unordered_set<T> const & arr) {
    os << '[';
    char const * sep = "";
    for (auto const & item: arr) {
      os << sep << item;
      sep = ", ";
    }
    os << ']';
    return os;
  }
  
  template <typename Ch, typename Tr, typename K, typename V>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, map<K, V> const & m) {
    os << '{';
    char const * sep = "";
    for (auto const & item: m) {
      os << sep << item.first << ": " << item.second;
      sep = ", ";
    }
    os << '}';
    return os;
  }

  template <typename Ch, typename Tr, typename K, typename V>
  basic_ostream<Ch, Tr> & operator<<(basic_ostream<Ch, Tr> & os, unordered_map<K, V> const & m) {
    os << '{';
    char const * sep = "";
    for (auto const & item: m) {
      os << sep << item.first << ": " << item.second;
      sep = ", ";
    }
    os << '}';
    return os;
  }

  template<typename T, size_t N> 
  struct hash<array<T, N>> {
    size_t operator()(array<T, N> const & key) const {
      hash<T> hasher;
      hash<size_t> sizehasher;
      size_t result = 0;
      for(size_t i = 0; i < N; ++i) {
        // result = (result << 1) ^ hasher(key[i]);
        result = sizehasher(result ^ hasher(key[i]));
      }
      return result;
    }
  };
} // namespace std


template <typename Iterator>
class StridedIterator {
public:
  // using ReturnType = decltype(*std::declval<Iterator>());
  // using ConstReturnType = decltype(*std::declval<Iterator const>());

  StridedIterator(Iterator && iterator, size_t stride)
  : _iterator(std::move(iterator)), _stride(stride) { }

  StridedIterator & operator++() {
    std::advance(_iterator, _stride);
    return *this;
  }
  StridedIterator operator++(int) {
    StridedIterator out = *this;
    std::advance(_iterator, _stride);
    return out;
  }
  auto & operator*() { return *_iterator; }
  auto const & operator*() const { return *_iterator; }

private:
  Iterator _iterator;
  size_t _stride;
};

