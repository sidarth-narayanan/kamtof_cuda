#include "gpu_api_functions.h"

namespace GDF
{

void set_gpu_global_local_range(const uint32_t gridDim, const uint32_t blockDim)
{
    assert(gridDim > 0 &&  blockDim > 0);
    gpu_manager->grid = {gridDim, 1, 1};
    gpu_manager->block = {blockDim, 1, 1};
    assert(blockDim % 32 == 0 || blockDim == 1);
    std::string message = "Note that you are setting a 1D Grid as {" + std::to_string(gpu_manager->grid.x) + ",1,1}"  +
                          " and 1D Block as {" + std::to_string(gpu_manager->block.x) + ",1,1}";
    log_msg(message);
}

// Allocates a gpu variable with given number of elements either in shared or device mode
template <bool is_malloc_shared /*= false*/>
void* malloc_gpu_var(const size_t& num_bytes)
{
    void* result = nullptr;
    if constexpr(is_malloc_shared)
        CUDA_CHECK(cudaMallocManaged(&result, num_bytes));
    else
        CUDA_CHECK(cudaMalloc(&result, num_bytes));
#ifdef GPU_MEM_LOG
    gpu_mem_map.insert({static_cast<void*>(result),num_bytes});
    tot_gpu_mem_used += num_bytes;
    CVG::string ptr_as_string;
    ptr_as_string.resize(100);
    sprintf(ptr_as_string.data(), "%p", result);
    CVG::string msg = "ptr = " + std::string(ptr_as_string.c_str()) + " | Allocated (MB)  = " + std::to_string(num_bytes/1e+06) + " | Running total memory usage(MB) = "
                      + std::to_string(tot_gpu_mem_used/1.0e+06);
    gpu_mem_usage_log << (tot_gpu_mem_used/1.0E+06) << "\n";
    log_msg(msg);
#endif
    return result;
}

template void* malloc_gpu_var<true>(const size_t& num_bytes);
template void* malloc_gpu_var<false>(const size_t& num_bytes);

static bool is_gpu_ptr(const void* const ptr)
{
    cudaPointerAttributes m_attr;
    CUDA_CHECK(cudaPointerGetAttributes(&m_attr, ptr));
    return !(m_attr.type == cudaMemoryType::cudaMemoryTypeUnregistered || m_attr.type == cudaMemoryType::cudaMemoryTypeHost);
}

// Copies the data from src to dest
void memcpy_gpu_var(void* const dest, const void * const src, const size_t& num_bytes)
{
    if(num_bytes > 0)
    {
        assert(src);
        assert(dest);

        if(is_gpu_ptr(src))
        {
            if(is_gpu_ptr(dest))
                gpu_manager->DtoD_memcpy_counter++;
            else
                gpu_manager->DtoH_memcpy_counter++;
        }
        else
        {
            if(is_gpu_ptr(dest))
                gpu_manager->HtoD_memcpy_counter++;
            else
                log_error("Trying to do a memcpy between two CPU pointers!!!");
        }

        CUDA_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyKind::cudaMemcpyDefault));
    }
}

// Free the gpu variable passed
void free_gpu_var(void* gpu_var)
{
    if(!gpu_var)
        return;
#ifdef GPU_MEM_LOG
    auto it = gpu_mem_map.find(gpu_var);
    if(it == gpu_mem_map.end())
    {
        log_msg<CDF::LogLevel::ERROR>("Error in free_gpu_var: GPU_MEM_MAP Look up failed!");
    }
    tot_gpu_mem_used -= it->second;
    CVG::string ptr_as_string;
    ptr_as_string.resize(100);
    sprintf(ptr_as_string.data(), "%p", gpu_var);
    std::string msg = "ptr = " + std::string(ptr_as_string.c_str()) + " | Released (MB)  = " + std::to_string(it->second/1e+06) +
                      " | Running total memory usage(MB) = " + std::to_string(tot_gpu_mem_used/1.0e+06);
    log_msg(msg);
    gpu_mem_usage_log << (tot_gpu_mem_used/1.0E+06) << "\n";
    gpu_mem_map.erase(gpu_var);
#endif
    assert(gpu_var);
    CUDA_CHECK(cudaFree(gpu_var));
}

// API Function for memeset
// 'val' is cast to unsigned char and asssigned to every byte
void memset_gpu_var(void* gpu_var, const int val, const size_t& num_bytes)
{
    assert(gpu_var);
    CUDA_CHECK(cudaMemset(gpu_var, val, num_bytes));
}

void transfer_to_gpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode)
{
   gpu_manager->transfer_to_gpu_internal(dsb_entry, transfer_mode);
}

void transfer_to_cpu(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode)
{
   gpu_manager->transfer_to_cpu_internal(dsb_entry, transfer_mode);
}

void transfer_all_silo_vars_to_cpu(const GDF::transfer_mode_t transfer_mode)
{
   // Lambda to bring back the SILO var
   auto m_transfer_to_cpu = [=](dataSetBase *entry) { GDF::transfer_to_cpu(entry, transfer_mode);};

   m_silo.for_each<CDF::StorageType::CELL>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::FACE>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::BOUNDARY>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::VECTOR>(m_transfer_to_cpu);
   m_silo.for_each<CDF::StorageType::PARAMETER>(m_transfer_to_cpu);
}

void copy_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("COPIED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(GDF::transfer_mode_t::COPY);
}

void move_all_silo_vars_to_cpu()
{
   log_msg<CDF::LogLevel::WARNING>("MOVED ALL SILO VARIBALES TO CPU");
   GDF::transfer_all_silo_vars_to_cpu(GDF::transfer_mode_t::MOVE);
}

} // namespace GDF
