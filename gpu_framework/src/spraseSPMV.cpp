#include "sparseSPMV.h"
#include "gpu_api_functions.h"

namespace GDF
{

sparseSPMV::sparseSPMV():
    alpha(0.0),
    beta(0.0),
    mat_op(CUSPARSE_OPERATION_NON_TRANSPOSE),
    alg(CUSPARSE_SPMV_ALG_DEFAULT),
    A(nullptr),
    x(nullptr),
    y(nullptr),
    m_handle(nullptr),
    workspace(nullptr)
{}

void sparseSPMV::init_system(const int64_t nrows, const int64_t ncols, const int64_t nnz, const double alpha_in, const double beta_in, int * const ia,
                             int * const ja, double * const matval, double * const vec, double * const result)
{
    // The system should not be setup before this
    assert(!is_setup());

    // Setup the cuSparse system
    CHECK_CUSPARSE(cusparseCreate(&m_handle));

    // Set the scalar coefficients
    alpha = alpha_in;
    beta = beta_in;

    CHECK_CUSPARSE(cusparseCreateCsr(&A, nrows, ncols, nnz, ia, ja, matval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Create and initialize dense vector handles
    CHECK_CUSPARSE(cusparseCreateDnVec(&x, ncols, vec, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&y, nrows, result, CUDA_R_64F));

    // Allocate external workspace
    std::size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(m_handle, mat_op, &alpha, A, x, &beta, y, CUDA_R_64F, alg, &buffer_size));
    workspace = GDF::malloc_gpu_var(buffer_size);

    // Optimize spmv
    CHECK_CUSPARSE(cusparseSpMV_preprocess(m_handle, mat_op, &alpha, A, x, &beta, y, CUDA_R_64F, alg, workspace));

    GDF::gpu_barrier();
}

void sparseSPMV::update_matrix(int * const ia, int * const ja, double * const matval)
{
    CHECK_CUSPARSE(cusparseCsrSetPointers(A, ia, ja, matval));
    GDF::gpu_barrier();
}

void sparseSPMV::update_x(double* const value)
{
    CHECK_CUSPARSE(cusparseDnVecSetValues(x, value));
    GDF::gpu_barrier();
}

void sparseSPMV::update_y(double* const value)
{
    CHECK_CUSPARSE(cusparseDnVecSetValues(y, value));
    GDF::gpu_barrier();
}

void sparseSPMV::compute()
{
    assert(A);
    assert(x);
    assert(y);
    assert(m_handle);

    // Perform the matrix multiplication
    CHECK_CUSPARSE(cusparseSpMV(m_handle, mat_op, &alpha, A, x, &beta, y, CUDA_R_64F, alg, workspace));
    GDF::gpu_barrier();
}

void sparseSPMV::release_system()
{
    assert(A);
    assert(x);
    assert(y);
    assert(m_handle);

    CHECK_CUSPARSE(cusparseDestroyDnVec(x));
    CHECK_CUSPARSE(cusparseDestroyDnVec(y));
    CHECK_CUSPARSE(cusparseDestroySpMat(A));
    CHECK_CUSPARSE(cusparseDestroy(m_handle));

    GDF::gpu_barrier();
    if(workspace) // Workspace can be nullptr for AMD gpus
        GDF::free_gpu_var(workspace);

    A = nullptr;
    x = nullptr;
    y = nullptr;
    workspace = nullptr;
    m_handle = nullptr;
}

} // namespace GDF
