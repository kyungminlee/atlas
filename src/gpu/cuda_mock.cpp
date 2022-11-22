#include <random>
#include <spdlog/spdlog.h>
#include "cuda_mock.hpp"

static thread_local int currentCudaDevice = 0;
std::mt19937 rng(0);

cudaError_t cudaSetDevice(int device)
{
    fmt::print("[{}] cudaSetDevice({})\n", currentCudaDevice, device);
    currentCudaDevice = device;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** p, size_t size)
{
    uintptr_t value = rng();
    uintptr_t * pi = reinterpret_cast<uintptr_t*>(p);
    *pi = value;
    fmt::print("[{}] cudaMalloc({} => {:#x}, {})\n", currentCudaDevice, reinterpret_cast<void const*>(p), *pi, size);
    return cudaSuccess;
}

cudaError_t cudaFree(void *p)
{
    fmt::print("[{}] cudaFree({})\n", currentCudaDevice, p);
    return cudaSuccess;
}
