#pragma once

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <numeric>
#include <iterator>

namespace atlas {


template <std::size_t D, typename I=std::size_t, typename C=I, typename M=I>
class multi_indexer;

template <std::size_t D, typename I=std::size_t, typename M=I>
class multi_index_const_iterator;


template <std::size_t D, typename I=std::size_t, typename M=I>
class multi_indexer
{
public:
	static_assert(D > 0, "dimension must be positive");

	static constexpr std::size_t dimension = D;
	using size_type = std::size_t;
	using index_type = I;
	using multi_index_type = M;
	
	using const_iterator = multi_index_const_iterator<D,I,M>;

	multi_indexer() = delete;
	multi_indexer(multi_indexer const &) = default;
	multi_indexer(multi_indexer &&) = default;
	multi_indexer & operator=(multi_indexer const &) = default;
	multi_indexer & operator=(multi_indexer &&) = default;

	multi_indexer(std::array<multi_index_type, dimension> const & shape, index_type offset = 0)
		: _lower({0,}), _upper(shape), _offset(offset)
	{
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (_upper[d] <= 0) {
				throw std::out_of_range("upper bound must be strictly larger than lower bound");
			}
			_stride[d+1] = _stride[d] * _upper[d];
			if (_stride[d+1] < _stride[d]) {
				throw std::overflow_error("Overflow in stride. index_type not large enough");
			}
		}
	}

	multi_indexer(
			std::array<multi_index_type, dimension> const & lower,
			std::array<multi_index_type, dimension> const & upper,
			index_type offset = 0
		)
		: _lower(lower), _upper(upper), _offset(offset)
	{
		_stride[0] = 1;
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (_upper[d] <= _lower[d]) {
				throw std::out_of_range("upper bound must be strictly larger than lower bound");
			}
			_stride[d+1] = _stride[d] * (_upper[d] - _lower[d]);
			if (_stride[d+1] < _stride[d]) {
				throw std::overflow_error("Overflow in stride. index_type not large enough");
			}
		}
	}

	multi_indexer(
			std::array<multi_index_type, dimension> const & lower, 
			std::array<multi_index_type, dimension> const & upper,
			std::array<index_type, dimension+1> const & stride,
			index_type offset = 0
	  )
		: _lower(lower), _upper(upper), _stride(stride), _offset(offset)
	{
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (_upper[d] <= _lower[d]) {
				throw std::out_of_range("upper bound must be strictly larger than lower bound");
			}
			// TODO: consider whether to allow this. (row-major allowance)
			if (_stride[d+1] < _stride[d] * (_upper[d] - _lower[d])) {
				throw std::out_of_range("stride smaller than minimum required");
			}
		}
	}


	/*
	multi_indexer slice(
			std::array<multi_index_type, dimension> const & lower,
			std::array<multi_index_type, dimension> const & upper
		)
	{
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (!(_lower[d] <= lower[d] && lower[d] <= upper[d] && upper[d] <= _upper[d])) {
				throw std::out_of_range("lower or upper out of range");
			}
		}
		return multi_indexer(lower, upper, _stride, _offset);
	}
	*/

	index_type ravel(std::array<multi_index_type, dimension> const & mindex) const {
		index_type idx = _offset;
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (mindex[d] < _lower[d]) { throw std::out_of_range("multi_index smaller than lower"); }
			if (mindex[d] >= _upper[d]) { throw std::out_of_range("multi_index greater than or equal to upper"); }
			idx += _stride[d] * (mindex[d] - _lower[d]);
		}
		return idx;
	}

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
		if (idx < _offset) {
			throw std::out_of_range("index cannot be smaller than offset");
		}
		idx -= _offset;
		if (idx >= _stride[dimension]) {
			throw std::out_of_range("index must be smaller than the upper stride + offset");
		}
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = static_cast<multi_index_type>((idx % _stride[d+1]) / _stride[d]) + _lower[d];
			if (out[d] >= _upper[d]) { throw std::out_of_range("out of range"); }
		}
		return out;
	}

	multi_index_type shape(size_t d) const { return _upper.at(d) - _lower[d]; }
	multi_index_type lower(size_t d) const { return _lower.at(d); }
	multi_index_type upper(size_t d) const { return _upper.at(d); }
	index_type stride(size_t d) const { return _stride.at(d); }

	std::array<multi_index_type, dimension> const & lower() const { return _lower; }
	std::array<multi_index_type, dimension> const & upper() const { return _upper; }
	std::array<index_type, dimension+1> const & stride() const { return _stride; }

	index_type offset() const { return _offset; }
	index_type size() const { return _stride[dimension]; }

	index_type begin_index() const { return offset(); }
	index_type end_index() const { return offset() + size(); }

	const_iterator begin() const { return const_iterator(this, begin_index()); }
	const_iterator end() const { return const_iterator(this, end_index()); }

	const_iterator cbegin() const { return const_iterator(this, begin_index()); }
	const_iterator cend() const { return const_iterator(this, end_index()); }

	/*
	bool increment(std::array<multi_index_type, dimension> & multi_index) const {
		if (multi_index[dimension-1] >= _upper[dimension-1]) {
			return false;
		}
		for (size_t d = 0 ; d < dimension ; ++d) {
			multi_index[d] += 1;
			if (multi_index[d] < _upper[d]) { // done
				return true;
			} else if (multi_index[d] == _upper[d]) { // carry
				if (d < dimension - 1) {
					multi_index[d] = _lower[d];
				} else {
					return true;
				}
			} else if (multi_index[d] > _upper[d]) {
				throw std::out_of_range("[multi_indexer::increment] multi_index out of range");
			}
		}
		throw std::out_of_range("[multi_indexer::increment] multi_index out of range overall");
		return false;
	}
	*/

private:
	std::array<multi_index_type, dimension> _lower;
	std::array<multi_index_type, dimension> _upper;
	std::array<index_type, dimension+1> _stride;
	index_type _offset;
};



template <std::size_t D, typename I=std::size_t, typename M=I>
class multi_index_const_iterator {
public:
	static constexpr std::size_t dimension = D;
	using size_type = std::size_t;
	using index_type = I;
	using multi_index_type = M;

	using difference_type = index_type;
	struct value_type {
		index_type index;
		std::array<multi_index_type> multi_index;
	};
	using pointer = void;
	using reference = value_type const &;

	const_iterator(): _parent(nullptr), _index(0) { }

	const_iterator(multi_indexer const * parent, index_type idx)
		: _parent(parent), _current(idx, parent->unravel(idx))
	{ }

	const_iterator(const_iterator const &) = default;
	const_iterator(const_iterator &&) = default;
	const_iterator & operator=(const_iterator const &) = default;

	~const_iterator() = default;
	
	reference operator*() const { return _current; }

	const_iterator& operator++() {
		++_current.index;
		_current.multi_index = _parent->unravel(_current.index);
		return *this;
	}

	const_iterator operator++(int) {
		const_iterator out(*this);
		++(*this);
		return out;
	}

	bool operator==(const_iterator const & rhs) const { return (_parent == rhs._parent) && (_index == rhs._index); }
	bool operator!=(const_iterator const & rhs) const { return (_parent != rhs._parent) || (_index != rhs._index); }

private:
	multi_indexer const * _parent;
	value_type _current;
};

} // namespace atlas
