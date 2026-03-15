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
   dataSetStorageGPU(const dataSetStorage<T, TYPE, DIMS>& dss_obj, const GDF::GPUManager_t* manager):
      dataSetGPU<T, DIMS>(), // The member are null/0 intialized here and will be set later in allocate_gpu_data_ptr() because we also need to set the statuses
      m_gpu_manager((assert(manager), manager)),
      cpu_dss_ptr(dss_obj.m_dsb()),
      m_name(dss_obj.name().c_str())
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
      m_gpu_manager(other.m_gpu_manager),
      cpu_dss_ptr(other.cpu_dss_ptr),
      m_name(other.m_name)
   {
#ifndef NDEBUG
      dataSetBaseGPU::is_read_only =other.is_read_only;
#endif
   }

   ~dataSetStorageGPU()
   {
      // By the time this destructor is called, the data members should already be cleaned up and set to nullptr

        assert(!this->m_gpu_data);
        assert(!this->m_offsets);
        assert(this->m_num_offsets == 0);
        assert(this->m_size == 0);
   }

   template <class... Indices>
   inline T& operator()(Indices&&... idx)
   {
      return dataSetGPU<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
   }

   template <class... Indices>
   const inline T& operator()(Indices&&... idx) const
   {
      return dataSetGPU<T,DIMS>::operator () (static_cast<Indices&&>(idx)...);
   }

   inline operator T& ()
   {
      static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
      return static_cast<T*>(dataSetBase::m_data)[0];
   }

   inline operator const T& () const
   {
      static_assert(TYPE == CDF::StorageType::PARAMETER && DIMS == 0, "This functionality is only supported for parameters");
      return static_cast<const T*>(dataSetBase::m_data)[0];
   }

   const GDF::GPUManager_t* get_manager() const
   {
      return m_gpu_manager;
   }

private:
   const GDF::GPUManager_t* m_gpu_manager = nullptr;

   /*
    * Stores a pointer of type DSS in DSB as the pointer object of this class is stored inside of GPUInstance
    * which in turn is held inside of StorageInfo which cannot be templated
    *
    * We would be returning this by casting it up using the templates of the GPU variable which might not be the same as the original CPU DSS's templates
   */
public:
   const dataSetBase* cpu_dss_ptr = nullptr;
   const char* m_name = nullptr;
};

#endif //DATASETSTORAGE_GPU_H
