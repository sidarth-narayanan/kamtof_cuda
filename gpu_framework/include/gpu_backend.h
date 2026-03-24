#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

#ifdef ENABLE_GPU
    #define gdf_kernel __host__ __device__
    #define gdf_device __device__
    #if defined(__CUDA_ARCH__)
        #define DEVICE_COMPILE
    #endif
#else
    #define gdf_kernel
    #define gdf_device
#endif


// From https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuSPARSE/bicgstab/bicgstab_example.c (Replaces return with std::exit())
#define CHECK_CUDA(func)                                                       \
{                                                                              \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n",         \
                   __LINE__, cudaGetErrorString(status), status);              \
            std::exit(EXIT_FAILURE);                                           \
    }                                                                          \
}
