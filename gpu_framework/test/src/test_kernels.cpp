#include <cmath> // For pow to take square
#include "test_kernels.h"
#include "gpu_globals.h" // For gpu_stream_data
#include "gpu_atomics.h" // For atomics
#include "gpu_api_functions.h"


gdf_kernel void kg_scale_pressure_and_change_scale::operator() (const size_t tid, const size_t stride) const
{
   if(tid < pressure.size())
   {
      pressure[tid] *= scale[0];
      if(tid == pressure.size()-1)
      {
         scale[0] = tid * 1.0;
      }
   }
}

gdf_kernel void kg_compute_temperature::operator() (const size_t tid, const size_t stride) const
{
   if(tid < temperature.size())
   {
      temperature[tid] = (pressure[tid] * volume[tid])/(n * R);
   }
}

gdf_kernel void kg_compute_pressure::operator() (const size_t tid, const size_t stride) const
{
   if(tid < pressure.size())
   {
      pressure[tid] = (n * R * temperature[tid]) / (volume[tid]);
   }
}

gdf_kernel void kg_test_kernel::operator() (const size_t tid, const size_t stride) const
{
   if(tid && x && y)
   {
      y = x;
   }
}

gdf_kernel void kg_set_initial_condition::operator() (const size_t tid, const size_t stride) const
{
   if(tid < velocity.size())
   {
      velocity(tid,0) = init_val_x;
      velocity(tid,1) = init_val_y;
      velocity(tid,2) = init_val_z;
   }
}

// void kg_norm2::operator() (nd_item<3> itm) const
// {
//    size_t idx = GDF::get_1d_index(itm);
//    size_t stride = GDF::get_1d_stride(itm);
// #ifndef DISABLE_GPU_KERNEL_ASSERTS
//    // This kernel works under the assumption that there is only one work-group
//    assert(itm.get_global_range(2) == itm.get_local_range(2));
// #endif
//    strict_fp_t sum = 0.0;
//    for(size_t ii = idx; ii < size; ii += stride)
//    {
//       sum += pow(vec[ii], 2.0);
//    }

//    if(idx < size)
//       vec[idx] = sum;

//    itm.barrier();

//    if(idx == 0)
//    {
//       sum = 0.0;
//       for(size_t ii = 0; ii < stride; ii++)
//       {
//          sum += vec[ii];
//       }
//       vec[1] = sum;
//       vec[0] = sqrt(sum);
//    }
// }

// void kg_atomics::operator() (nd_item<3> item) const
// {
//    size_t idx = GDF::get_1d_index(item);
//    size_t stride = GDF::get_1d_stride(item);
//    for(size_t ii = idx; ii < size; ii += stride)
//    {
//       GDF::atomic_add(result[0], vec[ii]);
//       GDF::atomic_sub(result[1], vec[ii]);
//       GDF::atomic_max(result[2], vec[ii]);
//       GDF::atomic_min(result[3], vec[ii]);
//    }
// }

// void kg_silo_null::operator ()(nd_item<3> item) const
// {
//    size_t idx = GDF::get_1d_index(item);
//    // Only do the subtraction if the GPU SILO object exist
//    if(gpu_silo_null.exists() && idx == gpu_random_idx)
//    {
//       gpu_silo_null[gpu_random_idx] -= gpu_subtract_val;
//    }
// }
