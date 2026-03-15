#pragma once

#include "gpu_api_functions.h"
#include "gpu_manager_t.h"
#include <cuda_runtime.h>

// Primary template: Default case, transfer_vars_to_gpu<uint8_t>() does not exist
template <typename, typename = std::void_t<>>
struct has_extractor : std::false_type {};

// Specialization: Detects if transfer_vars_to_gpu<uint8_t>() exists in the class
template <typename T>
struct has_extractor<T, std::void_t<decltype(std::declval<T>().template transfer_vars_to_gpu<1>())>> : std::true_type {};

// Function that calls transfer_vars_to_gpu<uint8_t>() only if it exists
template <typename Functor, typename... Us>
void callExtractorIfExists(Us&&... args)
{
    if constexpr (has_extractor<T>::value) // Compile-time check
    {
        constexpr uint8_t N = sizeof...(args);
        Functor obj{gpu_manager->extract_gpu_data_for_extractor(std::forward<Us>(args))...};
        obj.template transfer_vars_to_gpu<N>();
    }
}

template<typename Functor>
__global__ void submit_kernel(Functor& f)
{
    const size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const size_t stride = (GridDim.x * blockDim.x);

    f(tid, stride);
}

template<typename Functor, bool async = false, typename... Us>
void submit_to_gpu(Us&&... args)
{
    callExtractorIfExists<Functor>(args...);
    Functor obj{gpu_manager->extract_gpu_data_for_kernel(std::forward<Us>(args))...};

    submit_kernel<Functor><<<gpu_manager->grid, gpu_manager->block>>>(obj);

    if constexpr (!async)
        CUDA_CHECK(cudaDeviceSynchronize());
}
