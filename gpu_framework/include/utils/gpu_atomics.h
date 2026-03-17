#pragma once
#include "gpu_enums.h"

namespace GDF
{

template<class T>
gdf_device __forceinline__ void atomic_add(T& data, const T& value)
{
    atomicAdd(&data, value);
}

template<class T>
gdf_device __forceinline__ void atomic_sub(T& data, const T& value)
{
    atomicAdd(&data, -1.0*value);
}

template<class T>
gdf_device __forceinline__ void atomic_max(T& data, const T& value)
{
    atomicMax(&data, value);
}

template<class T>
gdf_device __forceinline__ void atomic_min(T& data, const T& value)
{
    atomicMin(&data, value);
}

template<>
gdf_device __forceinline__ void atomic_max(double& data, const double& val)
{
    unsigned long long int* addr = reinterpret_cast<unsigned long long int*>(&data);
    unsigned long long int old  = *addr, assumed;

    do
    {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;  // Current value already >= val, no update needed
        old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

template<>
gdf_device __forceinline__ void atomic_min(double& data, const double& val)
{
    unsigned long long int* addr = reinterpret_cast<unsigned long long int*>(&data);
    unsigned long long int old  = *addr, assumed;

    do
    {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) break;  // Current value already >= val, no update needed
        old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);
}


} // namespace GDF
