#pragma once

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <numeric>
#include <iterator>

#ifndef CHECK_RANGE
#ifndef NDEBUG
#define CHECK_RANGE
#endif
#endif

namespace atlas {

template <typename T> struct multi_indexer_trait;
template <typename Derive> class multi_indexer_base;
template <std::size_t D, typename I=std::size_t, typename M=I> class simple_multi_indexer;
template <typename Parent> class multi_indexer_slice;
template <typename Container> class multi_indexer_const_iterator;


template <typename Derived>
class multi_indexer_base
{
public:
	using derived_type = Derived;
	using trait_type = multi_indexer_trait<Derived>;

	using parent_type = typename trait_type::parent_type;
	static constexpr std::size_t dimension = trait_type::dimension;
	using size_type = typename trait_type::size_type;
	using index_type = typename trait_type::index_type;
	using multi_index_type = typename trait_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename trait_type::const_iterator;
	using slice_type = typename trait_type::slice_type;

	static_assert(dimension > 0, "dimension must be positive");

	multi_indexer_base() = default;

	slice_type slice(
		std::array<multi_index_type, dimension> const & l,
		std::array<multi_index_type, dimension> const & u		
	) const {
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (l[d] < lower(d)) { throw std::out_of_range("[multi_indexer_base.slice] slicing lower index cannot be smaller than the parent's lower index"); }
			if (u[d] <= l[d]) { throw std::out_of_range("[multi_indexer_base.slice] slicing lower index must be smaller than the slicing upper index"); }
			if (upper(d) < u[d]) { throw std::out_of_range("[multi_indexer_base.slice] slocing upper index cannot be greater than the parent's upper index"); }
		}
		return multi_indexer_slice<parent_type>(derived().parent(), l, u);
	}

	size_type ravel_compact(std::array<multi_index_type, dimension> const & mindex) const {
		size_type idx = 0;
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (mindex[d] < lower(d)) { throw std::out_of_range("multi_index smaller than lower"); }
			if (mindex[d] >= upper(d)) { throw std::out_of_range("multi_index greater than or equal to upper"); }
			idx += stride(d) * (mindex[d] - lower(d));
		}
		return idx;
	}

	std::array<multi_index_type, dimension> unravel_compact(size_type idx) const {
		if (idx >= size()) { throw std::out_of_range("compact index must be smaller than the size"); }
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = static_cast<multi_index_type>((idx % upper_stride(d)) / stride(d)) + lower(d);
			if (out[d] >= upper(d)) { throw std::out_of_range("out of range"); }
		}
		return out;
	}

	const_iterator begin() const { return const_iterator(reinterpret_cast<Derived const *>(this), 0); }
	const_iterator end() const { return const_iterator(reinterpret_cast<Derived const *>(this), size()); }

	const_iterator cbegin() const { return const_iterator(reinterpret_cast<Derived const *>(this), 0); }
	const_iterator cend() const { return const_iterator(reinterpret_cast<Derived const *>(this), size()); }

	size_type size(size_t d) const { return Derived::upper(d) - Derived::lower(d); }
	size_type shape(size_t d) const { return Derived::upper(d) - Derived::lower(d); }
	
	std::array<size_type, dimension> shape() const {
		std::array<size_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = size(d);
		}
		return out;
	}

	std::array<multi_index_type, dimension> lower() const {
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = lower(d);
		}
		return out;
	}

	std::array<multi_index_type, dimension> upper() const {
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = upper(d);
		}
		return out;
	}

	// Provided by the derived class
	parent_type const & parent() const { return derived().parent(); }
	multi_index_type lower(size_t d) const { return derived().lower(d); }
	multi_index_type upper(size_t d) const { return derived().upper(d); }
	size_type upper_stride(size_t d) const { return derived().upper_stride(d); }
	size_type stride(size_t d) const { return derived().stride(d); }
	size_type size() const { return derived().size(); }

	std::array<multi_index_type, dimension> unravel(index_type idx) const { return derived().unravel(idx); }
	index_type ravel(std::array<multi_index_type, dimension> const & midx) const { return derived().ravel(midx); }

private:
	Derived const & derived() const { return *reinterpret_cast<Derived const *>(this); }
};



template <std::size_t D, typename I, typename M>
class simple_multi_indexer : public multi_indexer_base<simple_multi_indexer<D, I, M>>
{
public:
	using trait_type = multi_indexer_trait<simple_multi_indexer>;

	using parent_type = typename trait_type::parent_type;
	static constexpr std::size_t dimension = trait_type::dimension;
	using size_type = typename trait_type::size_type;
	using index_type = typename trait_type::index_type;
	using multi_index_type = typename trait_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename trait_type::const_iterator;
	using slice_type = typename trait_type::slice_type;

	simple_multi_indexer() = delete;
	simple_multi_indexer(simple_multi_indexer const &) = default;
	simple_multi_indexer(simple_multi_indexer &&) = default;
	simple_multi_indexer & operator=(simple_multi_indexer const &) = default;
	simple_multi_indexer & operator=(simple_multi_indexer &&) = default;

	simple_multi_indexer(std::array<multi_index_type, dimension> const & shape)
		: _lower(), _upper(shape)
	{
		_stride[0] = 1;
		for (size_t d = 0 ; d < dimension ; ++d) {
			_lower[d] = 0;
			if (_upper[d] <= 0) {
				throw std::out_of_range("upper bound must be strictly larger than lower bound");
			}
			_stride[d+1] = _stride[d] * _upper[d];
			if (_stride[d+1] < _stride[d]) {
				throw std::overflow_error("Overflow in stride. index_type not large enough");
			}
		}
	}

	simple_multi_indexer(
			std::array<multi_index_type, dimension> const & lower,
			std::array<multi_index_type, dimension> const & upper
		)
		: _lower(lower), _upper(upper)
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

	parent_type const & parent() const { return *this; }
	multi_index_type lower(size_t d) const {
#ifdef CHECK_RANGE
		if (d >= dimension) { throw std::out_of_range("d must be smaller than dimension"); }
#endif
		return _lower[d];
	}
	multi_index_type upper(size_t d) const {
#ifdef CHECK_RANGE
		if (d >= dimension) { throw std::out_of_range("d must be smaller than dimension"); }
#endif
		return _upper[d];
	}
	size_type upper_stride(size_t d) const {
#ifdef CHECK_RANGE
		if (d >= dimension) { throw std::out_of_range("d must be smaller than dimension"); }
#endif
		return _stride[d+1];
	}	
	size_type stride(size_t d) const {
#ifdef CHECK_RANGE
		if (d >= dimension) { throw std::out_of_range("d must be smaller than dimension"); }
#endif
		return _stride[d];
	}
	size_type size() const { return _stride[dimension]; }

	index_type ravel(std::array<multi_index_type, dimension> const & mindex) const {
		index_type idx = 0;
		for (size_t d = 0 ; d < dimension ; ++d) {
#ifdef CHECK_RANGE
			if (mindex[d] < _lower[d]) { throw std::out_of_range("multi_index smaller than lower"); }
			if (mindex[d] >= _upper[d]) { throw std::out_of_range("multi_index greater than or equal to upper"); }
#endif
			idx += _stride[d] * (mindex[d] - _lower[d]);
		}
		return idx;
	}

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
#ifdef CHECK_RANGE
		if (idx >= _stride[dimension]) { throw std::out_of_range("index must be smaller than the upper stride + offset"); }
#endif
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = static_cast<multi_index_type>((idx % _stride[d+1]) / _stride[d]) + _lower[d];
#ifdef CHECK_RANGE
			if (out[d] >= _upper[d]) { throw std::out_of_range("out of range"); }
#endif
		}
		return out;
	}
private:
	std::array<multi_index_type, dimension> _lower;
	std::array<multi_index_type, dimension> _upper;
	std::array<index_type, dimension+1> _stride;
};


template <typename Parent>
class multi_indexer_slice : public multi_indexer_base<multi_indexer_slice<Parent>>
{
public:
	using trait_type = multi_indexer_trait<multi_indexer_slice>;

	using parent_type = typename trait_type::parent_type;
	static constexpr std::size_t dimension = trait_type::dimension;
	using size_type = typename trait_type::size_type;
	using index_type = typename trait_type::index_type;
	using multi_index_type = typename trait_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename trait_type::const_iterator;
	using slice_type = typename trait_type::slice_type;

	using compact_type = simple_multi_indexer<dimension, index_type, multi_index_type>;

	multi_indexer_slice() = delete;
	multi_indexer_slice(multi_indexer_slice const &) = default;
	multi_indexer_slice(multi_indexer_slice &&) = default;
	multi_indexer_slice & operator=(multi_indexer_slice const &) = default;
	multi_indexer_slice & operator=(multi_indexer_slice &&) = default;

	multi_indexer_slice(
			parent_type const & parent,
			std::array<multi_index_type, dimension> const & lower,
			std::array<multi_index_type, dimension> const & upper
		)
		: _parent(parent), _compact(lower, upper)
	{
#ifdef CHECK_RANGE
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (lower[d] < _parent.lower(d)) { throw std::out_of_range("[multi_indexer_slice] slicing lower index cannot be smaller than the parent's lower index"); }
			if (upper[d] <= lower[d]) { throw std::out_of_range("[multi_indexer_slice] slicing lower index must be smaller than the slicing upper index"); }
			if (_parent.upper(d) < upper[d]) { throw std::out_of_range("[multi_indexer_slice] slocing upper index cannot be greater than the parent's upper index"); }
		}
#endif
	}

	parent_type const & parent() const { return _parent; }
	multi_index_type lower(size_t d) const { return _compact.lower(d); }
	multi_index_type upper(size_t d) const { return _compact.upper(d); }
	size_type upper_stride(size_t d) const { return _compact.upper_stride(d); }
	size_type stride(size_t d) const { return _compact.stride(d); }
	size_type size() const { return _compact.size(); }

	index_type ravel(std::array<multi_index_type, dimension> const & midx) const {
#ifdef CHECK_RANGE
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (midx[d] < lower(d)) {
				throw std::out_of_range("multi index must be greater than or equal to the lower bound");
			} else if (midx[d] >= upper(d)) {
				throw std::out_of_range("multi index must be less than the upper bound");
			}
		}
#endif
		return _parent.ravel(midx);
	}

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
		auto midx = _parent.unravel(idx);
#ifdef CHECK_RANGE
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (midx[d] < lower(d)) {
				throw std::out_of_range("multi index must be greater than or equal to the lower bound");
			} else if (midx[d] >= upper(d)) {
				throw std::out_of_range("multi index must be less than the upper bound");
			}
		}
#endif
		return midx;
	}

private:
	parent_type _parent;
	compact_type _compact;
};


// Traits

template <size_t D, typename I, typename M>
struct multi_indexer_trait<simple_multi_indexer<D,I,M>> {
	static constexpr std::size_t dimension = D;
	using self_type = simple_multi_indexer<D, I, M>;
	using parent_type = self_type;
	using index_type = I;
	using multi_index_type = M;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using size_type = typename std::make_unsigned<index_type>::type;
	using const_iterator = multi_indexer_const_iterator<self_type>;
	using slice_type = multi_indexer_slice<self_type>;
};


template <size_t D, typename I, typename M>
struct multi_indexer_trait<multi_indexer_slice<simple_multi_indexer<D,I,M>>> {
	static constexpr std::size_t dimension = D;
	using self_type = multi_indexer_slice<simple_multi_indexer<D, I ,M>>;
	using parent_type = simple_multi_indexer<D, I, M>;
	using index_type = I;
	using multi_index_type = M;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using size_type = typename std::make_unsigned<index_type>::type;
	using const_iterator = multi_indexer_const_iterator<self_type>;
	using slice_type = self_type;
};



template <typename Container> // can be slice
class multi_indexer_const_iterator
{
public:
	using container_type = Container;
	using trait_type = multi_indexer_trait<container_type>;

	using parent_type = typename trait_type::parent_type;
	static constexpr std::size_t dimension = trait_type::dimension;
	using size_type = typename trait_type::size_type;
	using index_type = typename trait_type::index_type;
	using multi_index_type = typename trait_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename trait_type::const_iterator;

	struct value_type {
		size_type compact_index;
		index_type index;
		std::array<multi_index_type, dimension> multi_index;
		bool operator==(value_type const & rhs) const {
			return compact_index == rhs.compact_index;
		}
	};
	using difference_type = signed_index_type;
	using pointer = void;
	using reference = value_type const &;

	multi_indexer_const_iterator(): _container(nullptr), _current_compact_index(0) { }

	multi_indexer_const_iterator(container_type const * container, index_type cidx)
		: _container(container), _current_compact_index{cidx}
	{
	}

	multi_indexer_const_iterator(multi_indexer_const_iterator const &) = default;
	multi_indexer_const_iterator(multi_indexer_const_iterator &&) = default;
	multi_indexer_const_iterator & operator=(multi_indexer_const_iterator const &) = default;

	~multi_indexer_const_iterator() = default;
	
	value_type operator*() const {
		value_type out{_current_compact_index, 0, {}};
		out.multi_index = _container->unravel_compact(out.compact_index);
		out.index = _container->ravel(out.multi_index);
		return out;
	}

	multi_indexer_const_iterator& operator++() {
		++_current_compact_index;
		return *this;
	}

	multi_indexer_const_iterator& operator--() {
		--_current_compact_index;
		return *this;
	}

	multi_indexer_const_iterator operator++(int) {
		multi_indexer_const_iterator out(*this);
		++(*this);
		return out;
	}

	multi_indexer_const_iterator operator--(int) {
		multi_indexer_const_iterator out(*this);
		--(*this);
		return out;
	}

	bool operator==(multi_indexer_const_iterator const & rhs) const { return (_container == rhs._container) && (_current_compact_index == rhs._current_compact_index); }
	bool operator!=(multi_indexer_const_iterator const & rhs) const { return (_container != rhs._container) || (_current_compact_index != rhs._current_compact_index); }

private:
	container_type const * _container;
	index_type _current_compact_index;
	// value_type _current;
};

} // namespace atlas