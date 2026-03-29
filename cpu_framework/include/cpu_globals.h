#ifndef CPU_GLOBALS_H
#define CPU_GLOBALS_H

#ifndef CDF_GLOBAL
   #define CDF_GLOBAL extern
#endif

#include <set>
#include "fp_data_types.h"

CDF_GLOBAL int rank, numprocs;
CDF_GLOBAL int local_rank, local_numprocs;
CDF_GLOBAL bool gpu_solver;
CDF_GLOBAL bool implicit_solver;
CDF_GLOBAL strict_fp_t tol;
CDF_GLOBAL int tol_type;
CDF_GLOBAL int solver_type;
CDF_GLOBAL int num_iter;
CDF_GLOBAL size_t gpu_global_range;
CDF_GLOBAL size_t gpu_local_range;

// Global variable to tell if the pagefault call is triggered by the const (or) the non const operator of DSB
CDF_GLOBAL bool in_const_operator;

// Global variable to store the system page size (generally 4KB/ 4096B)
CDF_GLOBAL int system_page_size;

// Ordered set which stores the m_data value and it's corresponding DSB pointer in order to look up the DSB which triggered the pagefault
class dataSetBase;
CDF_GLOBAL std::set<std::pair<void*, dataSetBase*>> dsb_addr_set;

#endif // CPU_GLOBALS_H
