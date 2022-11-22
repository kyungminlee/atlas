#pragma once

#include <cstdlib>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

cudaError_t cudaSetDevice(int device);
cudaError_t cudaMalloc(void**p, size_t size);
cudaError_t cudaFree(void *p);
