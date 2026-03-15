#pragma once
#include "gpu_enums.h"

namespace GDF
{

template<class T>
gdf_kernel void atomic_add(T& data, const T& value)
{
}

template<class T>
gdf_kernel void atomic_sub(T& data, const T value)
{
}

template<class T>
gdf_kernel void atomic_max(T& data, const T value)
{
}

template<class T>
gdf_kernel void atomic_min(T& data, const T value)
{
}

} // namespace GDF
