#pragma once

#include <cstddef>

namespace atlas {

// trait type for indexing scheme
template <typename T> struct multi_indexer_traits;

// CRTP base classed used for all multi indexers
template <typename Derive> class multi_indexer_base;

// simple multi indexer (fortran style)
template <std::size_t D, typename I=std::size_t, typename M=I> class simple_multi_indexer;

// slice of an indexer. has local and global indexing different
template <typename Parent> class multi_indexer_slice;

// iterators for indexer
template <typename Container> class multi_indexer_const_iterator;

} // namespace atlas