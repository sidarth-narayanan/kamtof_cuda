#include "gpu_globals.h" // For GPU Globals
#include "gpu_api_functions.h" // For GPU API functions

// Allocate and initialize GPU variables
void setup_gpu_globals()
{
   #ifdef GPU_MEM_LOG
      tot_gpu_mem_used = 0;
      gpu_mem_usage_log.open("gpu_mem_usage.log", std::ios_base::trunc | std::ios_base::out);
   #endif

   // Setup GPU Manager (Has no variable which needs to be accesed on GPU)
   assert(!gpu_manager);
   gpu_manager = new GDF::GPUManager_t(gpu_global_range, gpu_local_range);

   init_cublashandle();

   log_msg("Device type selected : DEFAULT");
}

void finalize_gpu_globals()
{
   assert(gpu_manager);
   GDF::gpu_barrier(); // Wait for all the GPU related processes to end

   finalize_cublashandle();

   // Finalize GPU Manager
   delete gpu_manager;
   gpu_manager = nullptr;

   #ifdef GPU_MEM_LOG
      gpu_mem_usage_log.close();
   #endif
}
