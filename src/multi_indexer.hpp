#pragma once

#include <cstdlib>
#include <stdexcept>
#include <array>
#include <numeric>
#include <iterator>

#ifndef CHECK_RANGE
#  ifndef NDEBUG
#    define CHECK_RANGE 1
#  else
#    define CHECK_RANGE 0
#  endif
#endif

#if CHECK_RANGE
#define ASSERT_RANGE(p, msg) { if (!(p)) { throw std::out_of_range(msg); } } (void)0
#define ASSERT_OVERFLOW(p, msg) { if (!(p)) { throw std::overflow_error(msg); } } (void)0
#else
#define ASSERT_RANGE(p, msg) (void)0
#define ASSERT_OVERFLOW(p, msg) (void)0
#endif


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


template <typename Derived>
class multi_indexer_base
{
public:
	using derived_type = Derived;
	using traits_type = multi_indexer_traits<Derived>;

	using parent_type = typename traits_type::parent_type;
	static constexpr std::size_t dimension = traits_type::dimension;
	using size_type = typename traits_type::size_type;
	using index_type = typename traits_type::index_type;
	using multi_index_type = typename traits_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename traits_type::const_iterator;
	using slice_type = typename traits_type::slice_type;

	static_assert(dimension > 0, "dimension must be positive");

	multi_indexer_base() = default;

	slice_type slice(
		std::array<multi_index_type, dimension> l,
		std::array<multi_index_type, dimension> u		
	) const {
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(lower(d) <= l[d], "[multi_indexer_base.slice] slicing lower index cannot be smaller than the parent's lower index");
			ASSERT_RANGE(l[d] < u[d], "[multi_indexer_base.slice] slicing lower index must be smaller than the slicing upper index");
			ASSERT_RANGE(u[d] <= upper(d), "[multi_indexer_base.slice] slocing upper index cannot be greater than the parent's upper index");
		}
		return multi_indexer_slice<parent_type>(derived().parent(), l, u);
	}

	size_type ravel_compact(std::array<multi_index_type, dimension> const & mindex) const {
		size_type idx = 0;
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(lower(d) <= mindex[d], "multi_index smaller than lower");
			ASSERT_RANGE(mindex[d] < upper(d), "multi_index greater than or equal to upper");
			idx += stride(d) * (mindex[d] - lower(d));
		}
		return idx;
	}

	std::array<multi_index_type, dimension> unravel_compact(size_type idx) const {
		ASSERT_RANGE(idx < size(), "compact index must be smaller than the size");
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = static_cast<multi_index_type>((idx % upper_stride(d)) / stride(d)) + lower(d);
			ASSERT_RANGE(out[d] < upper(d), "out of range");
		}
		return out;
	}

	const_iterator begin() const { return const_iterator(reinterpret_cast<Derived const *>(this), 0); }
	const_iterator end() const { return const_iterator(reinterpret_cast<Derived const *>(this), size()); }

	const_iterator cbegin() const { return const_iterator(reinterpret_cast<Derived const *>(this), 0); }
	const_iterator cend() const { return const_iterator(reinterpret_cast<Derived const *>(this), size()); }

	size_type size(size_t d) const { return derived().upper(d) - derived().lower(d); }
	size_type shape(size_t d) const { return derived().upper(d) - derived().lower(d); }
	
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
	parent_type const & parent() const { return derived().parent(); } // for slicing
	multi_index_type lower(size_t d) const { return derived().lower(d); }
	multi_index_type upper(size_t d) const { return derived().upper(d); }
	size_type upper_stride(size_t d) const { return derived().upper_stride(d); }
	size_type stride(size_t d) const { return derived().stride(d); }
	size_type size() const { return derived().size(); }

	std::array<multi_index_type, dimension> unravel(index_type idx) const { return derived().unravel(idx); }
	index_type ravel(std::array<multi_index_type, dimension> midx) const { return derived().ravel(midx); }

private:
	Derived const & derived() const { return *reinterpret_cast<Derived const *>(this); }
};



//! simple_multi_indexer
//! 
//! represents Fortran-style column-major a contiguous indexing.
//!
//! row  0  1  2
//! col
//!  0   0  3  6
//!  1   1  4  7
//!  2   2  5  8
template <std::size_t D, typename I, typename M>
class simple_multi_indexer : public multi_indexer_base<simple_multi_indexer<D, I, M>>
{
public:
	using traits_type = multi_indexer_traits<simple_multi_indexer>;

	using self_type = typename traits_type::self_type;
	using base_type = typename traits_type::base_type;
	using parent_type = typename traits_type::parent_type;
	static constexpr std::size_t dimension = traits_type::dimension;
	using size_type = typename traits_type::size_type;
	using index_type = typename traits_type::index_type;
	using multi_index_type = typename traits_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename traits_type::const_iterator;
	using slice_type = typename traits_type::slice_type;

	simple_multi_indexer() = delete;
	simple_multi_indexer(simple_multi_indexer const &) = default;
	simple_multi_indexer(simple_multi_indexer &&) = default;
	simple_multi_indexer & operator=(simple_multi_indexer const &) = default;
	simple_multi_indexer & operator=(simple_multi_indexer &&) = default;

	explicit simple_multi_indexer(std::array<multi_index_type, dimension> shape)
		: _lower(), _upper(shape)
	{
		_stride[0] = 1;
		for (size_t d = 0 ; d < dimension ; ++d) {
			_lower[d] = 0;
			_stride[d+1] = _stride[d] * _upper[d];
			ASSERT_RANGE(0 < _upper[d], "upper bound must be strictly larger than lower bound");
			ASSERT_OVERFLOW(_stride[d] <= _stride[d+1], "Overflow in stride. index_type not large enough");
		}
	}

	// template <typename ... T, typename=typename std::enable_if<
	// 	sizeof...(T)==dimension*2
	// >::type>
	// simple_multi_indexer(T && ... args) {
	// 	using I = make_index_sequence<sizeof...(T)>;
	// }

	simple_multi_indexer(
			std::array<multi_index_type, dimension> lower,
			std::array<multi_index_type, dimension> upper
		)
		: _lower(lower), _upper(upper)
	{
		_stride[0] = 1;
		for (size_t d = 0 ; d < dimension ; ++d) {
			_stride[d+1] = _stride[d] * (_upper[d] - _lower[d]);
			ASSERT_RANGE(_lower[d] < _upper[d], "upper bound must be strictly larger than lower bound");
			ASSERT_OVERFLOW(_stride[d] <= _stride[d+1], "Overflow in stride. index_type not large enough");
		}
	}

	// simple_multi_indexer(std::initializer_list<multi_index_type> shape)
	// 	: _lower(), _upper()
	// {
	// 	ASSERT_RANGE(shape.size() != dimension, "dimension mismatch");
	// 	auto iter = shape.begin();
	// 	_stride[0] = 1;
	// 	for (size_t d = 0 ; d < dimension ; ++d) {
	// 		_lower[d] = 0;
	// 		_upper[d] = *iter++;
	// 		_stride[d+1] = _stride[d] * _upper[d];
	// 		ASSERT_RANGE(_upper[d] <= 0, "upper bound must be strictly larger than lower bound");
	// 		ASSERT_OVERFLOW(_stride[d+1] < _stride[d], "Overflow in stride. index_type not large enough");
	// 	}
	// }

	// simple_multi_indexer(
	// 		std::initializer_list<multi_index_type> lower,
	// 		std::initializer_list<multi_index_type> upper
	// 	)
	// 	: _lower(), _upper()
	// {
	// 	ASSERT_RANGE(lower.size() != dimension, "lower bound dimension mismatch");
	// 	ASSERT_RANGE(upper.size() != dimension, "upper bound dimension mismatch");
	// 	auto iter_lower = lower.begin();
	// 	auto iter_upper = upper.begin();
	// 	_stride[0] = 1;
	// 	for (size_t d = 0 ; d < dimension ; ++d) {
	// 		_lower[d] = *iter_lower++;
	// 		_upper[d] = *iter_upper++;
	// 		_stride[d+1] = _stride[d] * (_upper[d] - _lower[d]);
	// 		ASSERT_RANGE(_upper[d] <= _lower[d], "upper bound must be strictly larger than lower bound");
	// 		ASSERT_OVERFLOW(_stride[d+1] < _stride[d], "Overflow in stride. index_type not large enough");
	// 	}
	// }

	using base_type::slice;
	using base_type::ravel_compact;
	using base_type::unravel_compact;
	using base_type::begin;
	using base_type::end;
	using base_type::cbegin;
	using base_type::cend;
	using base_type::size;
	using base_type::shape;
	using base_type::lower;
	using base_type::upper;

	parent_type const & parent() const { return *this; }
	multi_index_type lower(size_t d) const {
		ASSERT_RANGE(d < dimension, "d must be smaller than dimension");
		return _lower[d];
	}
	multi_index_type upper(size_t d) const {
		ASSERT_RANGE(d < dimension, "d must be smaller than dimension");
		return _upper[d];
	}
	size_type upper_stride(size_t d) const {
		ASSERT_RANGE(d < dimension, "d must be smaller than dimension");
		return _stride[d+1];
	}	
	size_type stride(size_t d) const {
		ASSERT_RANGE(d < dimension, "d must be smaller than dimension");
		return _stride[d];
	}
	size_type size() const { return _stride[dimension]; }

	index_type ravel(std::array<multi_index_type, dimension> const & mindex) const {
		index_type idx = 0;
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(_lower[d] <= mindex[d], "multi_index smaller than lower");
			ASSERT_RANGE(mindex[d] < _upper[d], "multi_index greater than or equal to upper");
			idx += _stride[d] * (mindex[d] - _lower[d]);
		}
		return idx;
	}

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
		ASSERT_RANGE(idx < _stride[dimension], "index must be smaller than size");
		std::array<multi_index_type, dimension> out;
		for (size_t d = 0 ; d < dimension ; ++d) {
			out[d] = static_cast<multi_index_type>((idx % _stride[d+1]) / _stride[d]) + _lower[d];
			ASSERT_RANGE(out[d] < _upper[d], "out of range");
		}
		return out;
	}

	bool operator==(simple_multi_indexer const & rhs) const { return _lower == rhs._lower && _upper == rhs._upper; }
	bool operator!=(simple_multi_indexer const & rhs) const { return !( (*this) == rhs ); }

private:
	std::array<multi_index_type, dimension> _lower;
	std::array<multi_index_type, dimension> _upper;
	std::array<index_type, dimension+1> _stride;
};


template <typename Original>
class offset_multi_indexer : public multi_indexer_base<offset_multi_indexer<Original>>
{
public:
	using original_type = Original;

	using traits_type = multi_indexer_traits<offset_multi_indexer<original_type>>;

	using self_type = typename traits_type::self_type;
	using base_type = typename traits_type::base_type;
	using parent_type = typename traits_type::parent_type;
	static constexpr std::size_t dimension = traits_type::dimension;
	using size_type = typename traits_type::size_type;
	using index_type = typename traits_type::index_type;
	using multi_index_type = typename traits_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename traits_type::const_iterator;
	using slice_type = typename traits_type::slice_type;

	offset_multi_indexer(original_type const & original, index_type offset) : _original(original), _offset(offset) {
		ASSERT_OVERFLOW((*begin()).index <= (*begin()).index + size(), "index_type too small for the given offset and size");
	}

	using base_type::slice;
	using base_type::ravel_compact;
	using base_type::unravel_compact;
	using base_type::begin;
	using base_type::end;
	using base_type::cbegin;
	using base_type::cend;
	using base_type::size;
	using base_type::shape;
	using base_type::lower;
	using base_type::upper;

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
		ASSERT_RANGE(_offset <= idx, "index cannot be smaller than the offset");
		return _original.unravel(idx - _offset);
	}

	index_type ravel(std::array<multi_index_type, dimension> const & midx) const {
		return _original.ravel(midx) + _offset;
	}

	parent_type const & parent() const { return offset_multi_indexer(_original.parent()); }
	multi_index_type lower(size_t d) const { return _original.lower(d); }
	multi_index_type upper(size_t d) const { return _original.upper(d); }
	size_type upper_stride(size_t d) const { return _original.upper_stride(d); }
	size_type stride(size_t d) const { return _original.stride(d); }
	size_type size() const { return _original.size(); }

private:
	original_type const & _original;
	index_type _offset;
};


template <typename Parent>
class multi_indexer_slice : public multi_indexer_base<multi_indexer_slice<Parent>>
{
public:
	using traits_type = multi_indexer_traits<multi_indexer_slice>;

	using self_type = typename traits_type::self_type;
	using base_type = typename traits_type::base_type;
	using parent_type = typename traits_type::parent_type;
	static constexpr std::size_t dimension = traits_type::dimension;
	using size_type = typename traits_type::size_type;
	using index_type = typename traits_type::index_type;
	using multi_index_type = typename traits_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using unsigned_index_type = typename std::make_unsigned<index_type>::type;
	using unsigned_multi_index_type = typename std::make_unsigned<multi_index_type>::type;
	using const_iterator = typename traits_type::const_iterator;
	using slice_type = typename traits_type::slice_type;

	using compact_type = simple_multi_indexer<dimension, index_type, multi_index_type>;

	using base_type::slice;
	using base_type::ravel_compact;
	using base_type::unravel_compact;
	using base_type::begin;
	using base_type::end;
	using base_type::cbegin;
	using base_type::cend;
	using base_type::size;
	using base_type::shape;
	using base_type::lower;
	using base_type::upper;

	multi_indexer_slice() = delete;
	multi_indexer_slice(multi_indexer_slice const &) noexcept = default;
	multi_indexer_slice(multi_indexer_slice &&) noexcept = default;
	multi_indexer_slice & operator=(multi_indexer_slice const &) noexcept = default;
	multi_indexer_slice & operator=(multi_indexer_slice &&) noexcept = default;

	multi_indexer_slice(
			parent_type const & parent,
			std::array<multi_index_type, dimension> lower,
			std::array<multi_index_type, dimension> upper
		)
		: _parent(parent), _compact(lower, upper)
	{
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(_parent.lower(d) <= lower[d], "[multi_indexer_slice] slicing lower index cannot be smaller than the parent's lower index");
			ASSERT_RANGE(lower[d] < upper[d] , "[multi_indexer_slice] slicing lower index must be smaller than the slicing upper index");
			ASSERT_RANGE(upper[d] <= _parent.upper(d) , "[multi_indexer_slice] slocing upper index cannot be greater than the parent's upper index");
		}
	}

	parent_type const & parent() const { return _parent; }
	multi_index_type lower(size_t d) const { return _compact.lower(d); }
	multi_index_type upper(size_t d) const { return _compact.upper(d); }
	size_type upper_stride(size_t d) const { return _compact.upper_stride(d); }
	size_type stride(size_t d) const { return _compact.stride(d); }
	size_type size() const { return _compact.size(); }

	index_type ravel(std::array<multi_index_type, dimension> const & midx) const {
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(lower(d) <= midx[d] , "multi index must be greater than or equal to the lower bound");
			ASSERT_RANGE(midx[d] < upper(d), "multi index must be less than the upper bound");
		}
		return _parent.ravel(midx);
	}

	std::array<multi_index_type, dimension> unravel(index_type idx) const {
		auto midx = _parent.unravel(idx);
		for (size_t d = 0 ; d < dimension ; ++d) {
			ASSERT_RANGE(midx[d] >= lower(d), "multi index must be greater than or equal to the lower bound");
			ASSERT_RANGE(midx[d] < upper(d), "multi index must be less than the upper bound");
		}
		return midx;
	}

	// padding can only be positive
	multi_indexer_slice pad(unsigned_multi_index_type pad) const {
		// if (pad < 0) { throw std::out_of_range("padding should be positive only"); }
		std::array<multi_index_type, dimension> lower, upper;
		for (size_t d = 0 ; d < dimension ; ++d) {
			if (_compact.lower(d) - _parent.lower(d) <= pad) {
				lower[d] = _parent.lower(d);
			} else {
				lower[d] = _compact.lower(d) - pad;
			}
			if (_parent.upper(d) - _compact.upper(d) <= pad) {
				upper[d] = _parent.upper(d);
			} else {
				upper[d] = _compact.upper(d) + pad;
			}
		}
		return multi_indexer_slice(_parent, lower, upper);
	}

private:
	parent_type _parent;
	compact_type _compact;
};


// Traits

template <size_t D, typename I, typename M>
struct multi_indexer_traits<simple_multi_indexer<D,I,M>> {
	static constexpr std::size_t dimension = D;
	using self_type = simple_multi_indexer<D, I, M>;
	using base_type = multi_indexer_base<self_type>;
	using parent_type = self_type;
	using index_type = I;
	using multi_index_type = M;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using size_type = typename std::make_unsigned<index_type>::type;
	using const_iterator = multi_indexer_const_iterator<self_type>;
	using slice_type = multi_indexer_slice<self_type>;
};


// template <size_t D, typename I, typename M>
// struct multi_indexer_traits<multi_indexer_slice<simple_multi_indexer<D,I,M>>> {
// 	static constexpr std::size_t dimension = D;
// 	using self_type = multi_indexer_slice<simple_multi_indexer<D, I ,M>>;
// 	using base_type = multi_indexer_base<self_type>;
// 	using parent_type = simple_multi_indexer<D, I, M>;
// 	using index_type = I;
// 	using multi_index_type = M;
// 	using signed_index_type = typename std::make_signed<index_type>::type;
// 	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
// 	using size_type = typename std::make_unsigned<index_type>::type;
// 	using const_iterator = multi_indexer_const_iterator<self_type>;
// 	using slice_type = self_type;
// };



template <typename Parent>
struct multi_indexer_traits<multi_indexer_slice<Parent>> {
	static constexpr std::size_t dimension = Parent::dimension;
	using self_type = multi_indexer_slice<Parent>;
	using base_type = multi_indexer_base<self_type>;
	using parent_type = typename Parent::parent_type;
	using index_type = typename Parent::index_type;
	using multi_index_type = typename Parent::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using size_type = typename std::make_unsigned<index_type>::type;
	using const_iterator = multi_indexer_const_iterator<self_type>;
	using slice_type = self_type;
};


template <typename Original>
struct multi_indexer_traits<offset_multi_indexer<Original>> {
	static constexpr std::size_t dimension = Original::dimension;
	using self_type = offset_multi_indexer<Original>;
	using base_type = multi_indexer_base<self_type>;
	using parent_type = offset_multi_indexer<Original>;
	using index_type = typename Original::index_type;
	using multi_index_type = typename Original::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using size_type = typename std::make_unsigned<index_type>::type;
	using const_iterator = multi_indexer_const_iterator<self_type>;
	using slice_type = multi_indexer_slice<self_type>;
};


template <typename Container> // can be slice
class multi_indexer_const_iterator
{
public:
	using container_type = Container;
	using traits_type = multi_indexer_traits<container_type>;

	static constexpr std::size_t dimension = traits_type::dimension;
	using parent_type = typename traits_type::parent_type;
	using size_type = typename traits_type::size_type;
	using index_type = typename traits_type::index_type;
	using multi_index_type = typename traits_type::multi_index_type;
	using signed_index_type = typename std::make_signed<index_type>::type;
	using signed_multi_index_type = typename std::make_signed<multi_index_type>::type;
	using const_iterator = typename traits_type::const_iterator;

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

	value_type operator*() const {
		value_type out{_current_compact_index, 0, {}};
		out.multi_index = _container->unravel_compact(out.compact_index);
		out.index = _container->ravel(out.multi_index);
		return out;
	}

	value_type operator[](difference_type idx) const {
		value_type out{_current_compact_index + idx, 0, {}};
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

	multi_indexer_const_iterator & operator+=(difference_type n) {
		_current_compact_index += n;
		return *this;
	}
	multi_indexer_const_iterator & operator-=(difference_type n) {
		_current_compact_index -= n;
		return *this;
	}

	multi_indexer_const_iterator operator+(difference_type n) const {
		multi_indexer_const_iterator out(*this);
		out._current_compact_index = out._current_compact_index + n;
		return out;
	}

	multi_indexer_const_iterator operator-(difference_type n) const {
		multi_indexer_const_iterator out(*this);
		out._current_compact_index = out._current_compact_index - n;
		return out;
	}

	friend multi_indexer_const_iterator operator+(difference_type n, multi_indexer_const_iterator const & rhs) {
		multi_indexer_const_iterator out(rhs);
		out._current_compact_index = n + out._current_compact_index;
		return out;
	}

	difference_type operator-(multi_indexer_const_iterator const & rhs) const {
		if (_container != rhs._container) { return 0; }
		return _current_compact_index - rhs._current_compact_index;
	}

	bool operator==(multi_indexer_const_iterator const & rhs) const { return (_container == rhs._container) && (_current_compact_index == rhs._current_compact_index); }
	bool operator!=(multi_indexer_const_iterator const & rhs) const { return !(*this == rhs); }
	bool operator<(multi_indexer_const_iterator const & rhs) const {
		if (_container < rhs._container) { return true; }
		if (_container > rhs._container) { return false; }
		if (_current_compact_index < rhs._current_compact_index) { return true; }
		if (_current_compact_index > rhs._current_compact_index) { return false; }
		return false;
	}
	bool operator>(multi_indexer_const_iterator const & rhs) const { return rhs < *this; }
	bool operator>=(multi_indexer_const_iterator const & rhs) const { return !(*this < rhs); }
	bool operator<=(multi_indexer_const_iterator const & rhs) const { return !(rhs < *this); }

private:
	container_type const * _container;
	index_type _current_compact_index;
};

} // namespace atlasatlas