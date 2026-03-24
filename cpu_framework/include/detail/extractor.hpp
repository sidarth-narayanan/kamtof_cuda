#pragma once

#include "cpu_framework_enums.h"
#include "fp_data_types.h"

namespace  CDF
{

template<typename Type>
struct extractor_impl
{
   constexpr static POD_t PODType()
   {
      return POD_t::UNSUPPORTED_TYPE;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 0;
   }
};

template<>
struct extractor_impl<int8_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::INT8;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 1;
   }
};

template<>
struct extractor_impl<int16_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::INT16;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 2;
   }
};

template<>
struct extractor_impl<int32_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::INT32;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 4;
   }
};

template<>
struct extractor_impl<int64_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::INT64;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 8;
   }
};

template<>
struct extractor_impl<uint8_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::UINT8;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 8;
   }
};

template<>
struct extractor_impl<uint16_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::UINT16;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 2;
   }
};

template<>
struct extractor_impl<uint32_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::UINT32;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 4;
   }
};

template<>
struct extractor_impl<uint64_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::UINT64;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 8;
   }
};

template<>
struct extractor_impl<fp32_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::FP32;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 4;
   }
};

template<>
struct extractor_impl<fp64_t>
{
   constexpr static POD_t PODType()
   {
      return POD_t::FP64;
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return 8;
   }
};

template<typename Type>
struct extractor
{
   constexpr static POD_t PODType()
   {
      return extractor_impl<Type>::PODType();
   }

   constexpr static uint32_t single_element_byte_size()
   {
      return extractor_impl<Type>::single_element_byte_size();
   }

   friend struct extractor_impl<Type>;
};

} // namespace CDF
