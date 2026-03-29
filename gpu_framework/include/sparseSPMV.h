#pragma once

#include "cusparse_error.h"

/*
 * Perform sparse matrix - dense vector multiplication
 * y = (alpha * (mat_op(A) * x)) + (beta * y)
 * A -> Sparse matrix
 * x -> Dense vector to be multiplied with
 * y -> Solution dense vector
 * alpha,beta -> Scalar coefficients
 */

namespace GDF
{

struct sparseSPMV
{
    sparseSPMV();

    // Function to initialize the handles - does init, spmv_buffer_size and optimize
    void init_system(const int64_t nrows, const int64_t ncols, const int64_t nnz, const double alpha_in, const double beta_in, int* const ia,
                     int* const ja, double* const matval, double* const vec, double* const result);

    // Do the sparse matrix - dense vector multiplication
    void compute();

    // Reset the data
    void update_matrix(int * const ia, int * const ja, double * const matval);
    void update_x(double* const value);
    void update_y(double* const value);

    // Free/Finalize the
    void release_system();

    // Check to see if the system is already setup
    bool is_setup()
    {
        return m_handle;
    }

    // Scalar coefficients
    double alpha = 0.0;
    double beta = 0.0;

    // Sparse matrix data
    cusparseOperation_t mat_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;
    cusparseSpMatDescr_t A = nullptr;

    // Dense vector data
    cusparseDnVecDescr_t x = nullptr;
    cusparseDnVecDescr_t y = nullptr;

    // System description
    cusparseHandle_t m_handle = nullptr;

    // Temproary workspace
    void* workspace = nullptr;
};

} // namespace GDF
