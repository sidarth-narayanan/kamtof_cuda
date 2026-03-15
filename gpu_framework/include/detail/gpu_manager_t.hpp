#ifndef GPU_MANAGER_T_HPP
#define GPU_MANAGER_T_HPP

#include "gpu_manager_t.h"
#include "gpu_helpers.h" // For transfer_mode_to_cstr
#include "gpu_globals.h"
#include <type_traits>

namespace GDF
{

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void GPUManager_t::transfer_to_gpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                            transfer_mode_t transfer_mode /* = transfer_mode_t::NOT_SET */)
{
    GPUInstance_t& cur_gpu_instance= setup_gpu_instance_and_data_ptr(dss_obj);

    this->transfer_to_gpu_internal(cur_gpu_instance.get_cpu_dsb_ptr(), transfer_mode);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
void GPUManager_t::transfer_to_cpu_internal(const dataSetStorage<T, TYPE, DIMS>& dss_obj,
                                            transfer_mode_t transfer_mode /* = transfer_mode_t::MOVE */)
{
    const GPUInstance_t* const cur_gpu_instance = dss_obj.get_gpu_instance();
    if(cur_gpu_instance)
        this->transfer_to_cpu_internal(cur_gpu_instance->get_cpu_dsb_ptr(), transfer_mode);
}

/* In this function we take in and return an universal reference (&&) instead of a normal refernce (&)
 * as we need to handle both named variables (lvalues) and unnamed temproaries (rvalues).
*/
template <class  T>
T&& GPUManager_t::extract_gpu_data_for_kernel(T&& m_data)
{
    // Generic Function for named and unnamed variables of primitive types (e.g 'int', 'strict_fp_t')
    // We need the std::forward here to return an rvalue reference as an rvalue and reference and a lvalue reference as an lvalue reference
    return std::forward<T>(m_data);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPURead<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_kernel(dataSetStorageRead<T, TYPE, DIMS>& dss_obj)
{
#ifndef NDEBUG
    // If the variable does not exist, make sure the status is not allocated
    if(!dss_obj.exists())
    {
        assert(dss_obj.get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
    }
#endif

    // Check if the gpu_data is not OUT_OF_DATE (or) RESIZED_ON_CPU
    const xpu_data_status_t& gpu_data_status = dss_obj.get_xpu_data_status(xpu_t::GPU);
    if(gpu_data_status == xpu_data_status_t::OUT_OF_DATE || gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
    {
        std::string error = "Error in extract_gpu_data_for_kernel for SILO variable " + static_cast<std::string>(dss_obj.name()) + " : The gpu_data_status of the SILO variable is " + GDF::xpu_data_status_to_cstr(gpu_data_status) + ". It must either be UP_TO_DATE_READ (or) UP_TO_DATE_WRITE (or) TEMP_WRITE!";
        log_msg<CDF::LogLevel::ERROR>(error);
    }

    // This function specifically takes in a DSS object and returns respective the DSSGPU object
    assert(dss_obj.get_gpu_instance());
    dataSetBaseGPURead* dsb_gpu = dss_obj.get_gpu_instance()->get_gpu_dsb_ptr();
    assert(dsb_gpu);
    assert(dsb_gpu->void_data() || !dss_obj.exists() || (dss_obj.size() == 0)); // The data can be NULL either if the object doesn't exist (or) size is 0

    // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
    return *(static_cast<dataSetStorageGPURead<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPU<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_kernel(dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
#ifndef NDEBUG
    // If the variable does not exist, make sure the status is not allocated
    if(!dss_obj.exists())
    {
        assert(dss_obj.get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
    }
#endif

    // Check if the gpu_data is not OUT_OF_DATE (or) RESIZED_ON_CPU
    const xpu_data_status_t& gpu_data_status = dss_obj.get_xpu_data_status(xpu_t::GPU);
    if(gpu_data_status == xpu_data_status_t::OUT_OF_DATE || gpu_data_status == xpu_data_status_t::RESIZED_ON_CPU)
    {
        std::string error = "Error in extract_gpu_data_for_kernel for SILO variable " + static_cast<std::string>(dss_obj.name()) + " : The gpu_data_status of the SILO variable is " + GDF::xpu_data_status_to_cstr(gpu_data_status) + ". It must either be UP_TO_DATE_READ (or) UP_TO_DATE_WRITE (or) TEMP_WRITE!";
        log_msg<CDF::LogLevel::ERROR>(error);
    }

    // This function specifically takes in a DSS object and returns respective the DSSGPU object
    assert(dss_obj.get_gpu_instance());
    dataSetBaseGPU* dsb_gpu = dss_obj.get_gpu_instance()->get_gpu_dsb_ptr();
    assert(dsb_gpu);
    assert(dsb_gpu->void_data() || !dss_obj.exists() || (dss_obj.size() == 0)); // The data can be NULL either if the object doesn't exist (or) size is 0

    // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
    return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class  T>
T&& GPUManager_t::extract_gpu_data_for_extractor(T&& m_data)
{
    // Generic Function for named and unnamed variables of primitive types (e.g 'int', 'strict_fp_t')
    // We need the std::forward here to return an rvalue reference as an rvalue and reference and a lvalue reference as an lvalue reference
    return std::forward<T>(m_data);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPURead<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_extractor(dataSetStorageRead<T, TYPE, DIMS>& dss_obj)
{
    GPUInstance_t& cur_gpu_instance = setup_gpu_instance_and_data_ptr(dss_obj);
    dataSetBaseGPU* dsb_gpu = cur_gpu_instance.get_gpu_dsb_ptr();

    // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
    return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
dataSetStorageGPU<T, TYPE, DIMS>& GPUManager_t::extract_gpu_data_for_extractor(dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
    GPUInstance_t& cur_gpu_instance = setup_gpu_instance_and_data_ptr(dss_obj);
    dataSetBaseGPU* dsb_gpu = cur_gpu_instance.get_gpu_dsb_ptr();

    // We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
    return *(static_cast<dataSetStorageGPU<T, TYPE, DIMS>*>(dsb_gpu));
}

template<typename T, bool async, typename... Us>
void GPUManager_t::submit_to_gpu_internal(Us&&... args)
{
    callExtractorIfExists<T>(args...);

    // Call the actual kernel
    m_que.parallel_for(sycl::nd_range<3>{global_range,local_range}, T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});

    if constexpr (!async)
        m_que.wait();
}


template<typename T, typename... Us>
void GPUManager_t::submit_to_gpu_single_workgroup_internal(Us&&... args)
{
    // Call the actual kernel
    m_que.parallel_for(sycl::nd_range<3>{{1,1,SINGLE_WG_SIZE},{1,1,SINGLE_WG_SIZE}}, T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});
    m_que.wait();
}


template<typename T, typename... Us>
void GPUManager_t::single_task_gpu_internal(Us&&... args)
{
    // Call the actual kernel
    m_que.single_task(T{extract_gpu_data_for_kernel(std::forward<Us>(args))...});
    m_que.wait();
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /*= ZEROD*/>
void GPUManager_t::allocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
    GPUInstance_t* cur_gpu_instance = nullptr;
    if(!dss_obj.get_gpu_instance())
    {
        cur_gpu_instance = new GPUInstance_t(dss_obj.m_dsb());
        dss_obj.set_gpu_instance(cur_gpu_instance);
    }
    cur_gpu_instance = const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance());
    assert(cur_gpu_instance); // Should not be null
    if(!cur_gpu_instance->get_gpu_dsb_ptr()) // If the m_gpu_dss variable is not allocated, allocate that
    {
        /*
      *  Note that the members of DSBGPU is default intialized to null/0 values at this point and
      *  will only be populated correctly after the calling allocate_gpu_data_ptr()
      */
        dataSetStorageGPU<T, TYPE, DIMS>* dss_gpu_obj = new dataSetStorageGPU<T, TYPE, DIMS>(dss_obj, this);

        // DSSGPU pointer is stored as DSB pointer inside gpu_instance as it is not templated
        // Should cast up to DSSGPU when members/data of DSSGPU are accessed
        cur_gpu_instance->set_gpu_dsb_ptr(static_cast<dataSetBaseGPU*>(dss_gpu_obj));
    }
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /*= ZEROD*/>
void GPUManager_t::deallocate_gpu_instance(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
    assert(dss_obj.exists());
    GPUInstance_t* cur_gpu_instance= const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance());
    assert(cur_gpu_instance); // Should not be NULL
    assert(cur_gpu_instance->get_gpu_dsb_ptr()); // Should not be NULL

    delete cur_gpu_instance->get_gpu_dsb_ptr();
    cur_gpu_instance->set_gpu_dsb_ptr(nullptr);

    delete cur_gpu_instance;
    dss_obj.set_gpu_instance(nullptr);
}

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
GPUInstance_t& GPUManager_t::setup_gpu_instance_and_data_ptr(const dataSetStorage<T, TYPE, DIMS>& dss_obj)
{
    if(!dss_obj.get_gpu_instance())
    {
        allocate_gpu_instance(dss_obj);
    }
    GPUInstance_t& cur_gpu_instance= *(const_cast<GPUInstance_t*>(dss_obj.get_gpu_instance()));

    // If the CPU silo object doesn't exist, don't allocate gpu_data_ptr
    if(dss_obj.exists() && (cur_gpu_instance.get_xpu_data_status(xpu_t::GPU) == xpu_data_status_t::NOT_ALLOCATED))
    {
        allocate_gpu_data_ptr(&cur_gpu_instance, true);
    }
    return cur_gpu_instance;
}

} // namespace CDF

#endif // GPU_MANAGER_T_HPP
