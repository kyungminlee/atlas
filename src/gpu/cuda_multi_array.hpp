#include <memory>
#include <iostream>

#include "../multi_array.hpp"
#include "cuda_raw_ptr.hpp"

namespace atlas {

template <typename Value, typename Indexer>
struct multi_array_cuda;

template <typename Value, typename Indexer>
struct multi_array_view_cuda;

template <typename Value, typename Indexer>
struct multi_array_cuda : public multi_array_base<multi_array_cuda<Value, Indexer>>
{
    using value_type = Value;
    using indexer_type = Indexer;

    multi_array_cuda(int device, indexer_type const & indexer) noexcept(false)
    : _data(nullptr), _indexer(indexer)
    {
        cudaError_t cuda_stat;
        cuda_raw_ptr<value_type> p{device, nullptr};
        cuda_stat = cudaSetDevice(device);
        if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaSetDevice failed"); }
        cuda_stat = cudaMalloc(reinterpret_cast<void**>(&p.ptr), indexer.size() * sizeof(value_type));
        if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaMalloc failed"); }
        _data = std::unique_ptr<value_type[], cuda_deleter<value_type[]>>(p, cuda_deleter<value_type[]>()); // this is noexcept
    }

    multi_array_view_cuda<value_type, indexer_type> view() noexcept {
        return multi_array_view_cuda<value_type, indexer_type>(_data.get(), _indexer);
    }

    multi_array_view_cuda<value_type const, indexer_type> view() const noexcept {
        return multi_array_view_cuda<value_type const, indexer_type>(_data.get(), _indexer);
    }

    // operator multi_array_view_cuda<value_type, indexer_type> operator() noexcept {
    //     return this->view();
    // }

    // operator multi_array_view_cuda<value_type const, indexer_type> operator() const noexcept {
    //     return this->view();
    // }

    cuda_raw_ptr<value_type> data() { return _data.get(); }
    cuda_raw_ptr<value_type const> data() const { return _data.get(); }
    indexer_type const & indexer() const { return _indexer; }
private:
    std::unique_ptr<value_type[], cuda_deleter<value_type[]>> _data;
    indexer_type _indexer;
};

template <typename Value, typename Indexer>
multi_array_cuda<Value, Indexer> make_multi_array_cuda(int device, Indexer const & indexer)
{
    return multi_array_cuda<Value, Indexer>(device, indexer);
}


template <typename Value, typename Indexer>
struct multi_array_view_cuda
{
public:
    using value_type = Value;
    using indexer_type = Indexer;

    multi_array_view_cuda(value_type * data, indexer_type const & indexer) noexcept
    : _data(data), _indexer(indexer)
    {}

    value_type * data() noexcept { return _data; }
    value_type const * data() const noexcept { return _data; }
    indexer_type const & indexer() const { return _indexer; }
private:
    value_type * _data;
    indexer_type _indexer;    
};





template <typename Value, typename Indexer>
struct multi_array_traits<multi_array_cuda<Value, Indexer>>
{
    using value_type = Value;
    using indexer_type = Indexer;
    using self_type = multi_array_cuda<value_type, indexer_type>;

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

    using slice_type = basic_multi_array<value_type, typename indexer_type::slice_type>;
};



} // namespace atlas