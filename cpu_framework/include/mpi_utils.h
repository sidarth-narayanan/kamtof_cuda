#pragma once

#include "mpi.h"

void mpi_init(int* argc_ptr, char*** argv_ptr);

void mpi_finalize();

void mpi_abort();

void mpi_barrier();
