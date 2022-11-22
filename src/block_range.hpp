#pragma once

#include <iostream>

namespace atlas {

template <typename T> struct range_trait;
template <typename Range> struct range_iterator;
template <typename Value> struct block_range;

template <typename Value>
struct block_range {
  using traits_type = range_trait<block_range<Value>>;
  using size_type = typename traits_type::size_type;
  using value_type = typename traits_type::value_type;
  using const_iterator = typename traits_type::const_iterator;

  block_range(value_type begin, value_type end)
  : _begin(begin), _end(end) { }

  block_range(block_range const &) = default;
  block_range(block_range &&) = default;

  size_type size() const { return _end - _begin; }
  bool empty() const { return _end == _begin; }

  const_iterator begin() const { return const_iterator(_begin); }
  const_iterator end() const { return const_iterator(_end); }

  friend std::ostream & std::operator<<(std::ostream & os, block_range const &);

private:
  value_type _begin;
  value_type _end;
};


template <typename Value>
struct range_trait<block_range<Value>> {
  using size_type = std::size_t;
  using value_type = Value;
  using const_iterator = range_iterator<block_range<Value>>;
};


template <typename Range>
struct range_iterator {
  using traits_type = range_trait<Range>;
  using size_type = typename traits_type::size_type;
  using value_type = typename traits_type::value_type;
  using const_iterator = typename traits_type::const_iterator;
  using difference_type = typename std::make_signed<value_type>::type;

  range_iterator() {}
  range_iterator(value_type v): _current(v) { }
  range_iterator(range_iterator const &) = default;
  range_iterator(range_iterator &&) = default;
  range_iterator & operator=(range_iterator const &) = default;
  range_iterator & operator=(range_iterator &&) = default;

  operator value_type() const { return _current; }

  value_type operator*() const { return _current; }
  value_type operator[](difference_type n) const { return _current + n; }

  range_iterator & operator++() { ++_current; return *this; }
  range_iterator operator++(int) {
    range_iterator out(*this);
    ++_current;
    return out;
  }

  range_iterator & operator--() { --_current; return *this; }
  range_iterator operator--(int) {
    range_iterator out(*this);
    --_current;
    return out;
  }

  range_iterator operator+(difference_type n) const { return range_iterator(_current + n); }
  range_iterator operator-(difference_type n) const { return range_iterator(_current - n); }
	friend range_iterator operator+(difference_type n, range_iterator const & rhs) {
    return range_iterator(n + rhs._current);
  }

  difference_type operator-(range_iterator const & rhs) const {
    return static_cast<difference_type>(_current) - static_cast<difference_type>(rhs._current);
  }
  
  range_iterator & operator+=(difference_type n) { _current += n; return *this; }
  range_iterator & operator-=(difference_type n) { _current -= n; return *this; }

  bool operator==(range_iterator const & rhs) const { return _current == rhs._current; }
  bool operator!=(range_iterator const & rhs) const { return _current != rhs._current; }
  bool operator<(range_iterator const & rhs) const { return _current < rhs._current; }
	bool operator>(range_iterator const & rhs) const { return rhs < *this; }
	bool operator>=(range_iterator const & rhs) const { return !(*this < rhs); }
	bool operator<=(range_iterator const & rhs) const { return !(rhs < *this); }  
  value_type _current;
};



} // namespace atlas

namespace std {
  template<typename T>
  ostream & operator<<(ostream & os, atlas::block_range<T> const & r) {
    return os << '[' << r._begin << ", " << r._end << ')';
  }
}
