#pragma once

#include "datasetgpu.h"

template <class T, uint8_t DIMS /* = ZEROD */>
__host__ __device__ size_t dataSetGPU<T, DIMS>::byte_size() const
{
   return total_num_elements() * sizeof(T);
}

template <class T, uint8_t DIMS /* = ZEROD */>
__host__ __device__ size_t dataSetGPU<T, DIMS>::total_num_elements() const
{
   if constexpr (DIMS == ZEROD)
   {
      assert(!m_offsets);
      return this->m_size;
   }
   else
   {
      assert(m_offsets);
      return this->m_size*this->m_offsets[0];
   }
}
