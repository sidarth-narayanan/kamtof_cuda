#ifndef GPU_INSTANCE_T_H
#define GPU_INSTANCE_T_H

#include "gpu_enums.h" // For gpu enums xpu_data_status_t, transfer_mode_t and xpu_t
#include "gpu_silo_fwd.h" // For DSBGPU
#include "silo_fwd.h" // For DSB

namespace  GDF
{

class GPUInstance_t
{
public:
    GPUInstance_t(const dataSetBase* cpu_dsb);

    ~GPUInstance_t();

    const dataSetBaseGPU* get_gpu_dsb_ptr() const;

    dataSetBaseGPU* get_gpu_dsb_ptr();

    void set_gpu_dsb_ptr(dataSetBaseGPU* other);

    const xpu_data_status_t&  get_xpu_data_status(const xpu_t& device_type) const;

    void set_xpu_data_status(const xpu_t& device_type, const xpu_data_status_t& xpu_data_status);

    const transfer_mode_t& get_kernel_transfer_mode();

    void set_kernel_transfer_mode(const transfer_mode_t&);

    const dataSetBase *get_cpu_dsb_ptr() const; // Return the CPU DSB associated with this GPU instance

    void* gpu_data();

    // Stores a pointer of type DSSGPU in DSBGPU as the object of this class is stored inside of StorageInfo which cannot be templated
    dataSetBaseGPU* m_gpu_dss;
    transfer_mode_t kernel_transfer_mode = transfer_mode_t::NOT_SET;
    xpu_data_status_t cpu_data_status = xpu_data_status_t::UP_TO_DATE_WRITE;
    xpu_data_status_t gpu_data_status = xpu_data_status_t::NOT_ALLOCATED;
    dataSetBase* m_cpu_dsb = nullptr;
};

} // namespace GDF

#endif // GPU_INSTANCE_T_H
