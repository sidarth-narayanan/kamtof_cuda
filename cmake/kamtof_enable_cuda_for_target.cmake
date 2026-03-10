
include_guard(GLOBAL)

set( CXX_FLAGS_RELEASE " -O3 -DNDEBUG -Werror" )
set( CXX_FLAGS_DEBUG   "-g -O0 -fno-inline-functions -Wunused-result" )
set( CXX_FLAGS_ASAN    "-g -O3 -DNDEBUG -fsanitize=address -fno-omit-frame-pointer " )
set( CXX_FLAGS_RELWITHDEBINFO    "-g -O3 -DNDEBUG " )

if(ENABLE_GPU)
    set(NVCC_FLAGS_RELEASE "-O3")
    set(NVCC_FLAGS_DEBUG "-G -O0")
    set(NVCC_FLAGS_RELWITHDEBINFO "-O3 -lineinfo")

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_HOST_COMPILER ${CXX})
endif()
