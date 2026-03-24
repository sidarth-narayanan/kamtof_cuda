#pragma once

#include "gpu_manager_t.h" // For GPUManager_t functions
#include "gpu_globals.h" // For gpu_manager
#include "gpu_atomics.h" // For GPU atomics
#include "silo.h"
#include "datasetstoragegpu.h"
#include "gpu_backend.h"
#include "cublas_error.h"

namespace GDF
{

void set_gpu_global_local_range(const uint32_t gridDim, const uint32_t blockDim);

void transfer_to_gpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);

template <typename ... Types>
void transfer_to_gpu_move(const Types& ... dss_objs)
{
    // Fold expressions reduces (folds) a parameter pack over a binary operator (which is ',' here)
    (gpu_manager->transfer_to_gpu_internal(dss_objs, transfer_mode_t::MOVE), ...);
}

template <typename ... Types>
void transfer_to_gpu_readonly(const Types& ... dss_objs)
{
   (gpu_manager->transfer_to_gpu_internal(dss_objs, transfer_mode_t::READ_ONLY), ...);
}

template <typename ... Types>
void transfer_to_gpu_copy(const Types& ... dss_objs)
{
   (gpu_manager->transfer_to_gpu_internal(dss_objs, transfer_mode_t::COPY), ...);
}

template <typename ... Types>
void transfer_to_gpu_noinit(const Types& ... dss_objs)
{
   (gpu_manager->transfer_to_gpu_internal(dss_objs, transfer_mode_t::NOT_INITIALIZE), ...);
}

template <typename ... Types>
void transfer_to_gpu_syncandmove(const Types& ... dss_objs)
{
   (gpu_manager->transfer_to_gpu_internal(dss_objs, transfer_mode_t::SYNC_AND_MOVE), ...);
}

void transfer_to_cpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode);

template <typename ... Types>
void transfer_to_cpu_move(Types& ... dss_objs)
{
   (gpu_manager->transfer_to_cpu_internal(dss_objs, transfer_mode_t::MOVE), ...);
}

template <typename ... Types>
void transfer_to_cpu_readonly(Types& ... dss_objs)
{
   (gpu_manager->transfer_to_cpu_internal(dss_objs, transfer_mode_t::READ_ONLY), ...);
}

template <typename ... Types>
void transfer_to_cpu_copy(Types& ... dss_objs)
{
   (gpu_manager->transfer_to_cpu_internal(dss_objs, transfer_mode_t::COPY), ...);
}

template <typename ... Types>
void transfer_to_cpu_noinit(Types& ... dss_objs)
{
   (gpu_manager->transfer_to_cpu_internal(dss_objs, transfer_mode_t::NOT_INITIALIZE), ...);
}

template <typename ... Types>
void transfer_to_cpu_syncandmove(Types& ... dss_objs)
{
   (gpu_manager->transfer_to_cpu_internal(dss_objs, transfer_mode_t::SYNC_AND_MOVE), ...);
}

// Device wide barrier
inline void gpu_barrier()
{
    CHECK_CUDA(cudaDeviceSynchronize());
}

gdf_kernel inline void sync_threads()
{
#if defined(DEVICE_COMPILE)
    __syncthreads();
#endif
}

// Allocates a gpu variable with given number of elements either in shared or device mode
template<bool shared = false>
void* malloc_gpu_var(const size_t& num_bytes);

// Copies the data from src to dest
void memcpy_gpu_var(void* const dest, const void * const src, const size_t& num_bytes);

// Free the gpu variable passed
void free_gpu_var(void* gpu_var);

// API Function for memeset
// 'val' is cast to unsigned char and asssigned to every byte
void memset_gpu_var(void* gpu_var, const int val, const size_t& num_bytes);

template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
void memcpy_gpu_var(dataSetStorage<T, TYPE, DIMS>& dss_dest,const dataSetStorage <T, TYPE, DIMS>& dss_src)
{
    assert(dss_src.exists());
    assert(dss_dest.exists());
    assert((dss_src.byte_size() == dss_dest.byte_size()) && (dss_src.size() == dss_dest.size()));

    // If this object does not have a gpu_instance, create one by copying the older values from CPU
    const GPUInstance_t* const src_gpu_instance = dss_src.get_gpu_instance();
    if(!src_gpu_instance)
    {
        GDF::transfer_to_gpu_move(dss_src);
    }
    else
    {
        // If the GPU object already exist, make sure the data to be copied is not OUT_OF_DATE
        if(src_gpu_instance->get_xpu_data_status(GDF::xpu_t::GPU) != GDF::xpu_data_status_t::UP_TO_DATE_READ &&
            src_gpu_instance->get_xpu_data_status(GDF::xpu_t::GPU) != GDF::xpu_data_status_t::UP_TO_DATE_WRITE)
        {
            GDF::transfer_to_gpu_readonly(dss_src);
        }
    }
    assert(dss_src.get_gpu_instance());
    const T* const src_data = dss_src.gpu_data();
    assert(src_data);

    // Destination should be UP_TO_DATE_WRITE at the end of this
    GDF::transfer_to_gpu_noinit(dss_dest);
    assert(dss_dest.get_gpu_instance());
    T* const dest_data = dss_dest.gpu_data();
    assert(dest_data);

    GDF::memcpy_gpu_var(dest_data, src_data, dss_src.byte_size());
}

// GPU MATH API FUNCTIONS

void init_cublashandle();

void finalize_cublashandle();


class kg_axpby
{
public:
    kg_axpby(const uint64_t num_elements_in,
             const strict_fp_t a_in,
             const strict_fp_t* const x_in,
             const strict_fp_t b_in,
             const strict_fp_t* const y_in,
             strict_fp_t* const result_in):
        num_elements(num_elements_in),
        a(a_in),
        x(x_in),
        b(b_in),
        y(y_in),
        result(result_in)
    {}

    gdf_device void operator()(const size_t tid, const size_t stride) const;

private:
    const uint64_t num_elements;
    const strict_fp_t a;
    const strict_fp_t* const x;
    const strict_fp_t b;
    const strict_fp_t* const y;
    strict_fp_t* const result;
};

void axpy(const uint64_t num_elements, const strict_fp_t alpha, const strict_fp_t* const x, strict_fp_t* const y, uint8_t impl_type = 0);

void dot_product(const uint64_t num_elements, const strict_fp_t * const vec_a, const strict_fp_t * const vec_b, strict_fp_t *result, uint8_t impl_type = 0);

void linf_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void l0_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void l1_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result, uint8_t impl_type = 0);

void l2_norm(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result, uint8_t impl_type = 0);

void csr_matvec(const uint64_t nrow, const uint64_t ncol, const uint64_t nnz, const int* const ia, const int* const ja, const strict_fp_t * const matval, const
                strict_fp_t * const vec, strict_fp_t* const result, const int impl_type = 0);

void gpu_vec_sum(const uint64_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result);

void transfer_all_silo_vars_to_cpu(const GDF::transfer_mode_t transfer_mode);

void copy_all_silo_vars_to_cpu();

void move_all_silo_vars_to_cpu();

template <class  T>
void transfer_vars_to_gpu_internal(T&& m_data)
{
    return;
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void transfer_vars_to_gpu_internal(dataSetStorageGPU<T, TYPE, DIMS>& dss_obj)
{
    if(dss_obj.cpu_dss_ptr->get_xpu_data_status(xpu_t::GPU) != xpu_data_status_t::TEMP_WRITE)
        GDF::transfer_to_gpu(dss_obj.cpu_dss_ptr, transfer_mode_t::MOVE);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void transfer_vars_to_gpu_internal(const dataSetStorageGPU<T, TYPE, DIMS>& dss_obj)
{
    if(dss_obj.cpu_dss_ptr->get_xpu_data_status(xpu_t::GPU) != xpu_data_status_t::TEMP_WRITE)
        GDF::transfer_to_gpu(dss_obj.cpu_dss_ptr, transfer_mode_t::READ_ONLY);
}

template<uint8_t N, typename... Us>
void transfer_vars_to_gpu_impl(Us&&... args)
{
    static_assert(N ==  sizeof...(args));
    (transfer_vars_to_gpu_internal(args), ...);
}

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
    if constexpr (has_extractor<Functor>::value) // Compile-time check
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
    const size_t stride = (gridDim.x * blockDim.x);

    f(tid, stride);
}

template<typename Functor, bool async = false, typename... Us>
void submit_to_gpu_impl(Us&&... args)
{
    callExtractorIfExists<Functor>(args...);
    Functor obj{gpu_manager->extract_gpu_data_for_kernel(std::forward<Us>(args))...};

    submit_kernel<Functor><<<gpu_manager->grid, gpu_manager->block>>>(obj);

    if constexpr (!async)
        CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename Functor, bool async = false, typename... Us>
void submit_to_gpu_single_block_impl(Us&&... args)
{
    callExtractorIfExists<Functor>(args...);
    Functor obj{gpu_manager->extract_gpu_data_for_kernel(std::forward<Us>(args))...};

    submit_kernel<Functor><<<1, SINGLE_WG_SIZE>>>(obj);

    if constexpr (!async)
        CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename Functor, bool async = false, typename... Us>
void submit_single_impl(Us&&... args)
{
    callExtractorIfExists<Functor>(args...);
    Functor obj{gpu_manager->extract_gpu_data_for_kernel(std::forward<Us>(args))...};

    submit_kernel<Functor><<<1, 1>>>(obj);

    if constexpr (!async)
        CHECK_CUDA(cudaDeviceSynchronize());
}

template<typename Functor, typename... Us>
void submit_to_gpu(Us&&... args)
{
    submit_to_gpu_impl<Functor, false>(args...);
}

template<typename Functor, typename... Us>
void submit_to_gpu_async(Us&&... args)
{
    submit_to_gpu_impl<Functor, true>(args...);
}

template<typename Functor, typename... Us>
void submit_to_gpu_single_block(Us&&... args)
{
    submit_to_gpu_single_block_impl<Functor, false>(args...);
}

template<typename Functor, typename... Us>
void submit_to_gpu_single_block_async(Us&&... args)
{
    submit_to_gpu_single_block_impl<Functor, true>(args...);
}

template<typename Functor, typename... Us>
void submit_single(Us&&... args)
{
    submit_single_impl<Functor, false>(args...);
}

template<typename Functor, typename... Us>
void submit_single_async(Us&&... args)
{
    submit_single_impl<Functor, true>(args...);
}

} // namespace GDF
