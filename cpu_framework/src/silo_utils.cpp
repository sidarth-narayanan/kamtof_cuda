#include <string.h>

#include "silo_utils.h"
#include "cpu_framework_enums.h"
#include "logger.h"

std::string get_storage_type_name(const CDF::StorageType &TYPE)
{
   switch (TYPE)
   {
      case CDF::StorageType::CELL:
      {
         return "CELL";
         break;
      }
      case CDF::StorageType::FACE:
      {
         return "FACE";
         break;
      }
      case CDF::StorageType::BOUNDARY:
      {
         return "BOUNDARY";
         break;
      }
      case CDF::StorageType::VECTOR:
      {
         return "VECTOR";
         break;
      }
      case CDF::StorageType::PARAMETER:
      {
         return "PARAMETER";
         break;
      }
      default:
      {
         log_msg<CDF::LogLevel::ERROR>("Invaild Storagetype provided!");
         return "UNKNOWN_TYPE";
      }
   }
}

uint32_t get_single_element_byte_size(const CDF::POD_t &TYPE)
{
   switch (TYPE)
   {
      case CDF::POD_t::UINT8:
      {
         return 1;
         break;
      }
      case CDF::POD_t::UINT16:
      {
         return 2;
         break;
      }
      case CDF::POD_t::UINT32:
      {
         return 4;
         break;
      }
      case CDF::POD_t::UINT64:
      {
         return 8;
         break;
      }

      case CDF::POD_t::INT8:
      {
         return 1;
         break;
      }
      case CDF::POD_t::INT16:
      {
         return 2;
         break;
      }
      case CDF::POD_t::INT32:
      {
         return 4;
         break;
      }
      case CDF::POD_t::INT64:
      {
         return 8;
         break;
      }

      case CDF::POD_t::FP32:
      {
         return 4;
         break;
      }
      case CDF::POD_t::FP64:
      {
         return 8;
         break;
      }

      default:
      {
         log_msg<CDF::LogLevel::ERROR>("Invaild PODType provided!");
         return 0;
      }
   }
}
