#pragma once

#include "cusparse_v2.h"

// From https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuSPARSE/bicgstab/bicgstab_example.c (Replaces return with std::exit())
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf("cuSPARSE API failed at line %d with error: %s (%d)\n",     \
                   __LINE__, cusparseGetErrorString(status), status);          \
            std::exit(EXIT_FAILURE);                                           \
    }                                                                          \
}
