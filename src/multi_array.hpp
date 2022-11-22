#pragma once

#include <memory>
#include <utility>

#include "multi_indexer.hpp"

namespace atlas {

template<typename Array> struct multi_array_traits;
template<typename Derived> class multi_array_base;
template<typename Value, typename Indexer> class basic_multi_array;
template <typename Parent, typename Indexer> struct multi_array_view;

template <typename Value, size_t D, typename I=std::size_t, typename M=I>
using multi_array = basic_multi_array<Value, simple_multi_indexer<D, I, M>>;


template <typename Derived>
class multi_array_base
{
public:
    using self_type = Derived;
    using traits_type = multi_array_traits<Derived>;
    using value_type = typename traits_type::value_type;
    using indexer_type = typename traits_type::indexer_type;
    static constexpr size_t dimension = traits_type::dimension;
    using size_type = typename traits_type::size_type;
    using index_type = typename traits_type::index_type;
    using multi_index_type = typename traits_type::multi_index_type;
    using reference = typename traits_type::reference;
    using const_reference = typename traits_type::const_reference;

    using const_iterator = typename traits_type::const_iterator;
    using iterator = typename traits_type::iterator;

    operator Derived & () { return derived(); }
    operator Derived const & () const { return derived(); }

    // access by index
    const_reference operator[](index_type idx) const {
        return data()[indexer().ravel_compact(indexer().unravel(idx))];
    }

    reference operator[](index_type idx) {
        return data()[indexer().ravel_compact(indexer().unravel(idx))];
    }
    
    // access by multi index
    const_reference operator()(std::array<multi_index_type, dimension> midx) const {
        return data()[indexer().ravel_compact(midx)];
    }

    reference operator()(std::array<multi_index_type, dimension> midx) {
        return data()[indexer().ravel_compact(midx)];
    }

    size_type size() const { return indexer().size(); }

    // derived
    indexer_type const & indexer() const { return derived().indexer(); }
    value_type const * data() const noexcept { return derived().data(); }
    value_type * data() noexcept { return derived().data(); }

private:
    Derived const & derived() const { return *reinterpret_cast<Derived const *>(this); }
    Derived & derived() { return *reinterpret_cast<Derived *>(this); }
};


template<typename Value, typename Indexer>
class basic_multi_array : public multi_array_base<basic_multi_array<Value, Indexer>>
{
public:
    using value_type = Value;
    using indexer_type = Indexer;
    using self_type = basic_multi_array<Value, Indexer>;
    using base_type = multi_array_base<basic_multi_array<Value, Indexer>>;
    using traits_type = multi_array_traits<self_type>;
    static constexpr size_t dimension = traits_type::dimension;
    using size_type = typename traits_type::size_type;
    using index_type = typename traits_type::index_type;
    using multi_index_type = typename traits_type::multi_index_type;
    using reference = typename traits_type::reference;
    using const_reference = typename traits_type::const_reference;

    using const_iterator = typename traits_type::const_iterator;
    using iterator = typename traits_type::iterator;

    explicit basic_multi_array(Indexer const & indexer)
    : _indexer(indexer)
    , _data(_indexer.size())
    {}

    explicit basic_multi_array(std::array<multi_index_type, dimension> upper)
    : _indexer(upper)
    , _data(_indexer.size())
    {}

    basic_multi_array(
        std::array<multi_index_type, dimension> lower,
        std::array<multi_index_type, dimension> upper
        )
    : _indexer(lower, upper)
    , _data(_indexer.size())
    {}

    basic_multi_array(basic_multi_array const & rhs) = default;
    basic_multi_array(basic_multi_array && rhs) noexcept = default;

    // basic_multi_array & operator=(basic_multi_array const &) = default;
    // basic_multi_array & operator=(basic_multi_array && rhs) = default;

    iterator begin() noexcept { return std::begin(_data); }
    iterator end() noexcept { return std::begin(_data) + _indexer.size(); }
    const_iterator begin() const noexcept { return std::begin(_data); }
    const_iterator end() const noexcept { return std::begin(_data) + _indexer.size(); }
    const_iterator cbegin() const noexcept { return std::cbegin(_data); }
    const_iterator cend() const noexcept { return std::cbegin(_data) + _indexer.size(); }

    template<typename V, typename I>
    friend void std::swap(basic_multi_array<V, I> & lhs, basic_multi_array<V, I> & rhs) noexcept;

    basic_multi_array clone() const {
        basic_multi_array out(_indexer);
        std::copy(_data.begin(), _data.end(), out._data);
        return out;
    }

    using base_type::operator[];
    using base_type::operator();

    indexer_type const & indexer() const { return _indexer; }
    value_type const * data() const noexcept { return _data.data(); }
    value_type * data() noexcept { return _data.data(); }
private:
    indexer_type _indexer;
    std::vector<value_type> _data;
};


// template <typename Value, typename Indexer>
// struct multi_array_view : public multi_array_base
// {

// };


// traits

template <typename Value, typename Indexer>
struct multi_array_traits<basic_multi_array<Value, Indexer>>
{
    using value_type = Value;
    using indexer_type = Indexer;
    using self_type = basic_multi_array<value_type, indexer_type>;
    using parent_type = self_type;

    static constexpr size_t dimension = indexer_type::dimension;
    using size_type = typename indexer_type::size_type;
    using index_type = typename indexer_type::index_type;
    using multi_index_type = typename indexer_type::multi_index_type;

    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    using slice_type = multi_array_view<self_type, typename indexer_type::slice_type>;
};


template <typename Parent, typename Indexer>
struct multi_array_view
{
    using parent_type = Parent;
    using indexer_type = Indexer;
private:
    parent_type & _parent;
    indexer_type & _indexer;
};


// template <typename Parent, typename Indexer>
// struct multi_array_traits<multi_array_view<Parent, Indexer>>
// {
//     using value_type = typename multi_array_traits<Parent>::value_type;
//     using parent_type = typename Parent::parent_type;
//     using indexer_type = Indexer;
//     using self_type = basic_multi_array<value_type, indexer_type>;

//     static constexpr size_t dimension = indexer_type::dimension;
//     using size_type = typename indexer_type::size_type;
//     using index_type = typename indexer_type::index_type;
//     using multi_index_type = typename indexer_type::multi_index_type;

//     using reference = value_type &;
//     using const_reference = value_type const &;
//     using pointer = value_type *;
//     using const_pointer = value_type const *;
//     using iterator = typename std::vector<value_type>::iterator;
//     using const_iterator = typename std::vector<value_type>::const_iterator;

//     using slice_type = multi_array_view<parent_type, typename indexer_type::slice_type>;
// };

}


namespace std {

template<typename V, typename I>
void swap(atlas::basic_multi_array<V, I> & lhs, atlas::basic_multi_array<V, I> & rhs) noexcept {
    std::swap(lhs._indexer, rhs._indexer);
    std::swap(lhs._data, rhs._data);
}

}
