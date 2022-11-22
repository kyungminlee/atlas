#pragma once

#include "cuda_base.hpp"

namespace atlas {

// pointer type for cuda memory. Contains device information.
template <typename T>
struct cuda_raw_ptr {
    using element_type = T; 
    using difference_type = std::ptrdiff_t;

    cuda_raw_ptr() noexcept : device(-1), ptr(nullptr) { }
    cuda_raw_ptr(nullptr_t) noexcept : device(-1), ptr(nullptr) { }
    cuda_raw_ptr(int dev, T * p) noexcept : device(dev), ptr(p) { }
    cuda_raw_ptr(cuda_raw_ptr const &) noexcept = default;
    cuda_raw_ptr(cuda_raw_ptr &&) noexcept = default;

    cuda_raw_ptr & operator=(cuda_raw_ptr const &) noexcept = default;
    cuda_raw_ptr & operator=(cuda_raw_ptr &&) noexcept = default;
    cuda_raw_ptr & operator=(nullptr_t) noexcept {
        device = -1;
        ptr = nullptr;
        return *this;
    }

    operator T * () noexcept { return ptr; }
    operator T const * () const noexcept { return ptr; }
    operator cuda_raw_ptr<T const>() const noexcept { return cuda_raw_ptr<T const>{device, ptr}; }

    // TODO: check device?   
    cuda_raw_ptr operator+(difference_type d) const noexcept { return cuda_raw_ptr(device, ptr + d); }
    cuda_raw_ptr operator-(difference_type d) const noexcept { return cuda_raw_ptr(device, ptr - d); }
    difference_type operator-(cuda_raw_ptr const & rhs) const noexcept { return ptr - rhs.ptr; }

    cuda_raw_ptr & operator++() noexcept { ++ptr; return *this; }
    cuda_raw_ptr operator++(int) const noexcept {
        cuda_raw_ptr out = *this;
        ++(*this);
        return out;
    }
    cuda_raw_ptr & operator--() noexcept { --ptr; return *this; }
    cuda_raw_ptr operator--(int) const noexcept {
        cuda_raw_ptr out = *this;
        --(*this);
        return out;
    }

    cuda_raw_ptr & operator+=(difference_type d) noexcept {
        ptr += d;
        return *this;
    }

    cuda_raw_ptr & operator-=(difference_type d) noexcept {
        ptr -= d;
        return *this;
    }

    int device;
    T * ptr;
};


// Deleter for cuda memory
template <typename value_type>
class cuda_deleter
{
public:
    using pointer = cuda_raw_ptr<value_type>;

    void operator()(pointer p) const noexcept(false) {
        if (p.device >= 0 && p.ptr != nullptr) {
            cudaError_t cuda_stat;
            cuda_stat = cudaSetDevice(p.device);
            if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaSetDevice failed"); }
            cuda_stat = cudaFree(p.ptr);
            if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaFree failed"); }
        }
    }
};


template <typename value_type>
class cuda_deleter<value_type[]>
{
public:
    using pointer = cuda_raw_ptr<value_type>;

    void operator()(cuda_raw_ptr<value_type> p) const noexcept(false){
        if (p.device >= 0 && p.ptr != nullptr) {
            cudaError_t cuda_stat;
            cuda_stat = cudaSetDevice(p.device);
            if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaSetDevice failed"); }
            cuda_stat = cudaFree(p.ptr);
            if (cuda_stat != cudaSuccess) { throw cuda_exception("cudaFree failed"); }
        }
    }
};


} // namespace atlas