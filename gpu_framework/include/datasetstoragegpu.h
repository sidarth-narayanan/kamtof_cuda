#ifndef DATASETSTORAGE_GPU_H
#define DATASETSTORAGE_GPU_H

#include "silo_fwd.h" // For DSB and DSS
#include "datasetbasegpu.h" // For DSBGPU
#include "gpu_instance_t.h" // For GPUInstance member functions
#include "datasetstorage.h"
#include "datasetgpu.h"

namespace GDF
{
class GPUManager_t;
} // namespace GDF

template <class T, CDF::StorageType TYPE, uint8_t DIMS /* = ZEROD */>
class dataSetStorageGPU : public dataSetGPU<T, DIMS>
{
public:
    dataSetStorageGPU(const dataSetStorage<T, TYPE, DIMS>& dss_obj):
        dataSetGPU<T, DIMS>(), // The member are null/0 intialized here and will be set later in allocate_gpu_data_ptr() because we also need to set the statuses
        cpu_dss_ptr(dss_obj.m_dsb()),
        m_primary(true)
    {
#ifndef NDEBUG
        if(dss_obj.exists())
            assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::UP_TO_DATE_WRITE);
        else
            assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::CPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
        assert(cpu_dss_ptr->get_gpu_instance()->get_xpu_data_status(GDF::xpu_t::GPU) == GDF::xpu_data_status_t::NOT_ALLOCATED);
#endif
    }

    dataSetStorageGPU(const dataSetStorageGPU& other):
        dataSetGPU<T, DIMS>(other.m_gpu_data, other.m_num_offsets, other.m_offsets, other.m_size),
        cpu_dss_ptr(other.cpu_dss_ptr)
    {
#ifndef NDEBUG
        dataSetBaseGPU::is_read_only =other.is_read_only;
#endif
    }

    ~dataSetStorageGPU()
    {
        // By the time this destructor is called, the data members should already be cleaned up and set to nullptr
        if(m_primary)
        {
            assert(!this->m_gpu_data);
            assert(!this->m_offsets);
            assert(this->m_num_offsets == 0);
            assert(this->m_size == 0);
        }
    }

    template <class... Indices>
    gdf_kernel inline T& operator()(Indices&&... idx)
    {
        return dataSetGPU<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
    }

    template <class... Indices>
    gdf_kernel const inline T& operator()(Indices&&... idx) const
    {
        return dataSetGPU<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
    }

    gdf_kernel inline operator T& ()
    {
        static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
        return static_cast<T*>(dataSetBase::m_data)[0];
    }

    gdf_kernel inline operator const T& () const
    {
        static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
        return static_cast<const T*>(dataSetBase::m_data)[0];
    }

    const dataSetBase* cpu_dss_ptr = nullptr;
    const bool m_primary = false;
};

#endif //DATASETSTORAGE_GPU_H
