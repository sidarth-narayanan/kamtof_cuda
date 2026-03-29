#pragma once

#include <vector>
#include <array>
#include <mpi.h>
#include <algorithm>
#include <unordered_map>

#include "silo.h"
#include "silo_fwd.h"
#include "fp_data_types.h"
#include "input_struct.h"
#ifdef ENABLE_GPU
#include "gpu_api_functions.h"
#endif

class MpiClass 
{
public:
    int ssize;
    int rsize;
    int* slist;
    int* rlist;
    int* scounts;
    int* sdisp;
    int* rcounts;
    int* rdisp;
    strict_fp_t* sbuf;
    strict_fp_t* rbuf;

    int* gpu_slist;
    int* gpu_rlist;
    strict_fp_t* gpu_sbuf;
    strict_fp_t* gpu_rbuf;

    ~MpiClass()
    {
        delete[] slist;
        delete[] rlist;
        delete[] scounts;
        delete[] rcounts;
        delete[] sdisp;
        delete[] rdisp;
        delete[] sbuf;
        delete[] rbuf;

#ifdef ENABLE_GPU
        if(inputs->use_gpu_solver)
        {
            GDF::free_gpu_var(gpu_slist);
            GDF::free_gpu_var(gpu_rlist);
            GDF::free_gpu_var(gpu_sbuf);
            GDF::free_gpu_var(gpu_rbuf);
        }
#endif
    }
};

CDF_GLOBAL MpiClass* mpiparams;
