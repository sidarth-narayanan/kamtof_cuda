#include "gpu_instance_t.h" // For GPUInstance_t
#include <cassert> // For assert
#include "datasetbasegpu.h" // For DSBGPU members
#include "datasetbase.h"
#include "logger.hpp"

namespace  GDF
{

GPUInstance_t::GPUInstance_t(const dataSetBase* cpu_dsb)
{
    // Set the CPU and GPU DSB/DSS objects
    m_gpu_dss = nullptr;
    m_cpu_dsb = const_cast<dataSetBase*>(cpu_dsb);

    // Set the kernel and device data status values to their defaults
    kernel_transfer_mode = transfer_mode_t::NOT_SET;
    if(cpu_dsb->exists()) // If the parent dss does not exist, set the CPU data status to NOT_ALLOCATED. This is done to handle un-registered SILO vars
        cpu_data_status = xpu_data_status_t::UP_TO_DATE_WRITE;
    else
        cpu_data_status = xpu_data_status_t::NOT_ALLOCATED;
    gpu_data_status = xpu_data_status_t::NOT_ALLOCATED;
}

GPUInstance_t::~GPUInstance_t()
{
    // m_gpu_dss should be destroyed and set to NULL by GPUManager_t::deallocate_gpu_instance()
    assert(!m_gpu_dss);
}

void* GPUInstance_t::gpu_data()
{
    return m_gpu_dss->void_data();
}

const dataSetBaseGPU* GPUInstance_t::get_gpu_dsb_ptr() const
{
    return m_gpu_dss;
}

dataSetBaseGPU* GPUInstance_t::get_gpu_dsb_ptr()
{
    return m_gpu_dss;
}

void GPUInstance_t::set_gpu_dsb_ptr(dataSetBaseGPU* other)
{
    m_gpu_dss = other;
}

const xpu_data_status_t &GPUInstance_t::get_xpu_data_status(const xpu_t &device_type) const
{
    if(device_type == xpu_t::CPU)
    {
        return cpu_data_status;
    }
    else if(device_type == xpu_t::GPU)
    {
        return gpu_data_status;
    }
    else
    {
        log_msg<CDF::LogLevel::ERROR>("Invalid device type passed to function get_xpu_data_status. The device_type can only be xpu_t::CPU (or) xpu_t::GPU");
        return cpu_data_status;
    }
}

void GPUInstance_t::set_xpu_data_status(const xpu_t& device_type, const xpu_data_status_t& xpu_data_status)
{
    if(device_type == xpu_t::CPU)
    {
        cpu_data_status = xpu_data_status;
    }
    else if(device_type == xpu_t::GPU)
    {
        gpu_data_status = xpu_data_status;
    }
    else
    {
        log_msg<CDF::LogLevel::ERROR>("Invalid device type passed to function get_xpu_data_status. The device_type can only be xpu_t::CPU (or) xpu_t::GPU");
    }
}

const transfer_mode_t& GPUInstance_t::get_kernel_transfer_mode()
{
    return kernel_transfer_mode;
}

void GPUInstance_t::set_kernel_transfer_mode(const transfer_mode_t& kernel_transfer_mode_in)
{
    kernel_transfer_mode = kernel_transfer_mode_in;
}

const dataSetBase* GPUInstance_t::get_cpu_dsb_ptr() const
{
    return m_cpu_dsb;
}

} // namespace GDF
