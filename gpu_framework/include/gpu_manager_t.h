#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "silo_fwd.h" // For DSB and DSS
#include "datasetstoragegpu.h" // For DSSGPU, SILO error handling, GPUInstance member functions
#include "gpu_silo_fwd.h" // For DSSGPURead
#include "logger.h"
#include "cpu_globals.h"
#include "gpu_backend.h"

namespace GDF
{

class GPUManager_t
{
public:
   /* Using "const uint64_t (&glob_range)[3]" to clarify to the compiler that we want a reference to an array,
    * rather than the (invalid) array of references (i.e. uint64_t & glob_range[3])
    *
    * Also, this only accepts an array of 3 integers, rather than an array of arbitrary size (which will happen if we use "uint64_t glob_range[3]")
   */
   GPUManager_t(const uint32_t gridDim, const uint32_t blockDim):
      grid{gridDim, 1, 1}, block{blockDim, 1, 1},
      HtoD_memcpy_counter(0), DtoH_memcpy_counter(0), DtoD_memcpy_counter(0)
   {
       int DevCount = -1;
       CHECK_CUDA(cudaGetDeviceCount(&DevCount));
       assert(DevCount >= 0);
       if(local_numprocs > DevCount)
           log_msg("Number of ranks > number of GPUs!!!!!!");
       int m_dev = local_rank % DevCount;
       CHECK_CUDA(cudaSetDevice(m_dev))
       CHECK_CUDA(cudaGetDeviceProperties(&m_prop, m_dev))

       m_device_name = m_prop.name;

      // We currently only support 1D grids and blocks
      assert(grid.y == 1);
      assert(grid.z == 1);
      assert(block.y == 1);
      assert(block.z == 1);
      assert(block.x <= 1024);

      std::string message = "Device Set || Name: " + m_device_name;
      log_msg(message);

      std::string grid_msg = "Total number of threads launched for the GPU solver : " + std::to_string(grid.x * block.x);
      std::string block_msg = "Number of threads launched per workgroup for the GPU solver : " + std::to_string(block.x);
      log_msg(grid_msg);
      log_msg(block_msg);
   }

   ~GPUManager_t()
   {
       print_gpu_memcpy_counts();
   }

   template <class  T>
   T&& extract_gpu_data_for_kernel(T&& m_data);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   dataSetStorageGPURead<T, TYPE, DIMS>& extract_gpu_data_for_kernel(dataSetStorageRead<T, TYPE, DIMS>& dss_obj);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   dataSetStorageGPU<T, TYPE, DIMS>& extract_gpu_data_for_kernel(dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <typename T, typename... Us>
   void callExtractorIfExists(Us&&... args);

   template <class T>
   T&& extract_gpu_data_for_extractor(T&& m_data);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   dataSetStorageGPU<T, TYPE, DIMS>& extract_gpu_data_for_extractor(dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   dataSetStorageGPURead<T, TYPE, DIMS>& extract_gpu_data_for_extractor(dataSetStorageRead<T, TYPE, DIMS>& dss_obj);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   bool check_gpu_offsets_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class  T,  CDF::StorageType  TYPE,  uint8_t DIMS = ZEROD>
   void update_gpu_offsets_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void allocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void deallocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   GPUInstance_t& setup_gpu_instance_and_data_ptr(const dataSetStorage<T, TYPE, DIMS>& dss_obj);

   void allocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance, const bool set_offsets);

   void deallocate_gpu_data_ptr(GPUInstance_t* cur_gpu_instance);

   void resize_gpu_data_ptr(const dataSetBase* const dsb_obj);

   void transfer_to_gpu_internal(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode = transfer_mode_t::NOT_SET);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void transfer_to_gpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                 transfer_mode_t transfer_mode = transfer_mode_t::NOT_SET);

   template <class T, CDF::StorageType TYPE, uint8_t DIMS = ZEROD>
   void transfer_to_cpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                 transfer_mode_t transfer_mode = transfer_mode_t::MOVE);

   void transfer_to_cpu_internal(const dataSetBase* const dsb_entry, transfer_mode_t transfer_mode = transfer_mode_t::MOVE);

   void print_gpu_memcpy_counts();

   dim3 grid;
   dim3 block;

   // Device Properties
   std::string m_device_name;
   cudaDeviceProp m_prop;

   size_t HtoD_memcpy_counter;
   size_t DtoH_memcpy_counter;
   size_t DtoD_memcpy_counter;
};

}  // namespace GDF

#include "gpu_manager_t.hpp"
