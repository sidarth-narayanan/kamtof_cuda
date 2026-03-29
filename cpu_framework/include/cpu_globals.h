#pragma once

#ifndef CDF_GLOBAL
    #define CDF_GLOBAL extern
#endif

#include <set>

// MPI Globals
CDF_GLOBAL int rank, numprocs;
CDF_GLOBAL int local_rank, local_numprocs;

// Global variable to tell if the pagefault call is triggered by the const (or) the non const operator of DSB
CDF_GLOBAL bool in_const_operator;

// Global variable to store the system page size (generally 4KB/ 4096B)
CDF_GLOBAL int system_page_size;

// Ordered set which stores the m_data value and it's corresponding DSB pointer in order to look up the DSB which triggered the pagefault
class dataSetBase;
CDF_GLOBAL std::set<std::pair<void*, dataSetBase*>> dsb_addr_set;
