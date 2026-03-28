#include "gpu_api_functions.h" // For GPU API functions
#include "gpu_atomics.h" // For GPU Atomics
#include "sparseSPMV.h"

#include "cublas_error.h"

static cublasHandle_t m_handle;

namespace GDF
{

void init_cublashandle()
{
    CHECK_CUBLAS(cublasCreate(&m_handle))
}

void finalize_cublashandle()
{
    CHECK_CUBLAS(cublasDestroy(m_handle))
}


__device__ void kg_axpby::operator()(const size_t tid, const size_t stride) const
{
    for(int ii = tid; ii < num_elements; ii += stride)
    {
        result[ii] = (a * x[ii]) + (b * y[ii]);
    }
}

void axpy(const size_t num_elements, const strict_fp_t alpha, const strict_fp_t* const x, strict_fp_t* const y, uint8_t impl_type /* = 0 */)
{
    assert(x);
    assert(y);

    switch (impl_type)
    {
    case 0:
    {
        CHECK_CUBLAS(cublasDaxpy_64(m_handle, num_elements, &alpha, x, 1, y, 1))
        GDF::gpu_barrier();
        break;
    }
    case 1:
    {
        GDF::submit_to_gpu<kg_axpby>(num_elements, alpha, x, 1.0, y, y);
        break;
    }
    default:
    {
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::axpy : Incorrect version number provided!");
    }
    }
}

class kg_dot_product_single_workgroup
{
public:
    kg_dot_product_single_workgroup(const size_t& num_elements_in, const strict_fp_t * const vec_a_in, const strict_fp_t * const vec_b_in,
                                    strict_fp_t* const thread_result_in, strict_fp_t* const result_in):
        num_elements(num_elements_in),
        vec_a(vec_a_in),
        vec_b(vec_b_in),
        thread_result(thread_result_in),
        result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        strict_fp_t local_result = 0.0;
        for(size_t kk = tid; kk < num_elements; kk += stride)
        {
            local_result += (vec_a[kk] * vec_b[kk]);
        }
        thread_result[tid] = local_result;
        sync_threads();
        if(tid == 0)
        {
            result[GZ] = 0.0;
            for(size_t ii = 0; ii < stride; ii++)
            {
                result[GZ] += thread_result[ii];
            }
        }
    }

private:
    const size_t num_elements;
    const strict_fp_t* const vec_a;
    const strict_fp_t* const vec_b;
    strict_fp_t* const thread_result;
    strict_fp_t* const result;
};

void dot_product(const size_t num_elements, const strict_fp_t * const vec_a, const strict_fp_t * const vec_b, strict_fp_t *result, uint8_t impl_type /* = 0 */)
{
    assert(vec_a);
    assert(vec_b);
    assert(result);

    switch (impl_type)
    {
    case 0:
    {
        CHECK_CUBLAS(cublasDdot_64(m_handle, num_elements, vec_a, 1, vec_b, 1, result))
        GDF::gpu_barrier();
        break;
    }
    case 1:
    {
        strict_fp_t* thread_result = static_cast<strict_fp_t*>(GDF::malloc_gpu_var(sizeof(strict_fp_t)*SINGLE_WG_SIZE));
        GDF::submit_to_gpu_single_block<kg_dot_product_single_workgroup>(num_elements, vec_a, vec_b, thread_result, result);
        GDF::free_gpu_var(thread_result);
        break;
    }
    default:
    {
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::dot_product : Incorrect version number provided!");
    }
    }
}

class kg_linf_norm_single_workgroup
{
public:
    kg_linf_norm_single_workgroup(const size_t num_elements_in, const strict_fp_t* const vec_in, strict_fp_t* const thread_result_in, strict_fp_t* const result_in):
        num_elements(num_elements_in),
        vec(vec_in),
        thread_result(thread_result_in),
        result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        thread_result[tid] = -1.0* std::numeric_limits<strict_fp_t>::max();
        for(size_t ii = tid; ii < num_elements; ii += stride)
        {
            thread_result[tid] = fmax(thread_result[tid], fabs(vec[ii]));
        }
        sync_threads();
        if(tid == 0)
        {
            result[GZ] = -1.0* std::numeric_limits<strict_fp_t>::max();
            for(size_t ii = 0; ii < stride; ii++)
            {
                result[GZ] = fmax(result[GZ], fabs(thread_result[ii]));
            }
        }
    }

private:
    const size_t num_elements;
    const strict_fp_t* const vec;
    strict_fp_t* const thread_result;
    strict_fp_t* const result;
};

void linf_norm(const size_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result)
{
    assert(vec);
    assert(result);
    strict_fp_t* thread_result = static_cast<strict_fp_t*>(GDF::malloc_gpu_var(sizeof(strict_fp_t) * SINGLE_WG_SIZE));

    GDF::memset_gpu_var(thread_result, 0, sizeof(strict_fp_t)*SINGLE_WG_SIZE);

    GDF::submit_to_gpu_single_block<kg_linf_norm_single_workgroup>(num_elements, vec, thread_result, result);

    GDF::free_gpu_var(thread_result);
}

class kg_l0_norm_single_workgroup
{
public:
    kg_l0_norm_single_workgroup(const size_t num_elements_in, const strict_fp_t* const vec_in, size_t* const thread_result_in, strict_fp_t* const result_in):
        num_elements(num_elements_in),
        vec(vec_in),
        thread_result(thread_result_in),
        result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        for(size_t ii = tid; ii < num_elements; ii += stride)
        {
            if(vec[ii] != 0.0)
                thread_result[tid]++;
        }
        sync_threads();
        if(tid == 0)
        {
            result[GZ] = 0.0;
            for(size_t ii = 0; ii < stride; ii++)
            {
                result[GZ] += thread_result[ii];
            }
        }
    }

private:
    const size_t num_elements;
    const strict_fp_t* const vec;
    size_t* const thread_result;
    strict_fp_t* const result;
};

void l0_norm(const size_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result)
{
    assert(vec);
    assert(result);
    size_t* thread_result = static_cast<size_t*>(GDF::malloc_gpu_var(sizeof(size_t) * SINGLE_WG_SIZE));

    GDF::memset_gpu_var(thread_result, 0, sizeof(strict_fp_t) * SINGLE_WG_SIZE);

    GDF::submit_to_gpu_single_block<kg_l0_norm_single_workgroup>(num_elements, vec, thread_result, result);

    GDF::free_gpu_var(thread_result);
}

class kg_l1_norm_single_workgroup
{
public:
    kg_l1_norm_single_workgroup(const size_t num_elements_in, const strict_fp_t* const vec_in, strict_fp_t* const thread_result_in, strict_fp_t* const result_in):
        num_elements(num_elements_in),
        vec(vec_in),
        thread_result(thread_result_in),
        result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        for(size_t ii = tid; ii < num_elements; ii += stride)
        {
            thread_result[tid] += fabs(vec[ii]);
        }
        sync_threads();
        if(tid == 0)
        {
            result[GZ] = 0.0;
            for(size_t ii = 0; ii < stride; ii++)
            {
                result[GZ] += thread_result[ii];
            }
        }
    }

private:
    const size_t num_elements;
    const strict_fp_t* const vec;
    strict_fp_t* const thread_result;
    strict_fp_t* const result;
};

void l1_norm(const size_t num_elements, const strict_fp_t* const vec, strict_fp_t * const result, uint8_t impl_type /* = 0 */)
{
    assert(vec);
    assert(result);

    switch (impl_type)
    {
    case 0:
    {
        CHECK_CUBLAS(cublasDasum_64(m_handle, num_elements, vec, 1, result))
        GDF::gpu_barrier();
        break;
    }
    case 1:
    {
        strict_fp_t* thread_result = static_cast<strict_fp_t*>(GDF::malloc_gpu_var(sizeof(strict_fp_t) * SINGLE_WG_SIZE));
        GDF::submit_to_gpu_single_block<kg_l1_norm_single_workgroup>(num_elements, vec, thread_result, result);
        GDF::free_gpu_var(thread_result);
        break;
    }
    default:
    {
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::linf_norm: Incorrect version number provided!");
    }
    }
}

class kg_sqrt_single_task
{
public:
    kg_sqrt_single_task(strict_fp_t* const result_in):result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        result[GZ] = sqrt(result[GZ]);
    }

private:
    strict_fp_t* const result;
};

void l2_norm(const size_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result, uint8_t impl_type /* = 0 */)
{
    assert(vec);
    assert(result);

    switch (impl_type)
    {
    case 0:
    {
        CHECK_CUBLAS(cublasDnrm2_64(m_handle, num_elements, vec, 1, result))
        GDF::gpu_barrier();
        break;
    }
    case 1:
    {
        GDF::dot_product(num_elements, vec, vec, result);
        GDF::submit_single<kg_sqrt_single_task>(result);
        break;
    }
    default:
    {
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::l2_norm: Incorrect version number provided!");
    }
    }

    GDF::gpu_barrier();
}

class kg_csr_matvec
{
public:
    kg_csr_matvec(const int* const ia_in, const int* const ja_in, const strict_fp_t* const matval_in, const strict_fp_t* const vec_in,
                  strict_fp_t* const result_in, const size_t& n_in):
        ia(ia_in),
        ja(ja_in),
        matval(matval_in),
        vec(vec_in),
        result(result_in),
        n(n_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        for(size_t kk = tid; kk < n; kk += stride)
        {
            result[kk] = 0.0;
            for(int jj = ia[kk]; jj < ia[kk+1]; jj++)
            {
                result[kk] += (matval[jj] * vec[ja[jj]]);
            }
        }
    }

private:
    const int* const ia;
    const int* const ja;
    const strict_fp_t* const matval;
    const strict_fp_t* const vec;
    strict_fp_t* const result;
    const size_t n;
};

void csr_matvec(const size_t nrow, const size_t ncol, const size_t nnz, const int* const ia, const int* const ja, const strict_fp_t * const matval, const
                strict_fp_t * const vec, strict_fp_t* const result, const int impl_type /* = 0 */)
{
    assert(ia);
    assert(ja);
    assert(matval);
    assert(vec);
    assert(result);

    switch (impl_type)
    {
    case 0:
    {
        int* const ia_new = const_cast<int*>(ia);
        int* const ja_new = const_cast<int*>(ja);
        strict_fp_t* const matval_new = const_cast<strict_fp_t*>(matval);
        strict_fp_t* const vec_new = const_cast<strict_fp_t*>(vec);
        sparseSPMV m_sys;
        m_sys.init_system(nrow, ncol, nnz, 1.0, 0.0, ia_new, ja_new, matval_new, vec_new, result);
        m_sys.compute();
        m_sys.release_system();
        break;
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::csr_matvec: cuSparse isn't available for SpMv yet!");
        break;
    }
    case 1:
    {
        assert(nrow == ncol);
        GDF::submit_to_gpu<kg_csr_matvec>(ia, ja, matval, vec, result, nrow);
        break;
    }
    default:
    {
        log_msg<CDF::LogLevel::ERROR>("Error in GDF::csr_matvec: Incorrect version number provided!");
    }
    }
}

class kg_vec_sum_single_workgroup
{
public:
    kg_vec_sum_single_workgroup(const size_t num_elements_in, const strict_fp_t* const vec_in, strict_fp_t* const thread_result_in, strict_fp_t* const result_in):
        num_elements(num_elements_in),
        vec(vec_in),
        thread_result(thread_result_in),
        result(result_in)
    {}

    __device__ void operator()(const size_t tid, const size_t stride) const
    {
        for(size_t ii = tid; ii < num_elements; ii += stride)
        {
            thread_result[tid] += vec[ii];
        }
        sync_threads();
        if(tid == 0)
        {
            result[GZ] = 0.0;
            for(size_t ii = 0; ii < stride; ii++)
            {
                result[GZ] += thread_result[ii];
            }
        }
    }

private:
    const size_t num_elements;
    const strict_fp_t* const vec;
    strict_fp_t* const thread_result;
    strict_fp_t* const result;
};

void gpu_vec_sum(const size_t num_elements, const strict_fp_t* const vec, strict_fp_t* const result)
{
    assert(vec);
    assert(result);
    strict_fp_t* thread_result = static_cast<strict_fp_t*>(GDF::malloc_gpu_var(sizeof(strict_fp_t) * SINGLE_WG_SIZE));

    GDF::memset_gpu_var(thread_result, 0, sizeof(strict_fp_t) * SINGLE_WG_SIZE);

    GDF::submit_to_gpu_single_block<kg_vec_sum_single_workgroup>(num_elements, vec, thread_result, result);

    GDF::free_gpu_var(thread_result);
}

} // namespace GDF
