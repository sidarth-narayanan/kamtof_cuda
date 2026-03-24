#pragma once

#include "cublas_v2.h"

// From https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuSPARSE/bicgstab/bicgstab_example.c (Replaces return with std::exit())
#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
        cublasStatus_t status = (func);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            printf("CUBLAS API failed at line %d with error: %d\n",            \
                   __LINE__, status);                                          \
            std::exit(EXIT_FAILURE);                                           \
    }                                                                          \
}
