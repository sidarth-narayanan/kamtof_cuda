cmake_minimum_required( VERSION 3.20 )

project( "GPU Framework" )

set( GPU_FRAMEWORK_DIR  $ENV{GPU_FRAMEWORK_DIR} )

aux_source_directory( ${GPU_FRAMEWORK_DIR}/src GpuFrameworkSrcTop)

if(ENABLE_GPU)
   set_source_files_properties(${GpuFrameworkSrcTop} PROPERTIES LANGUAGE CUDA)
endif()

add_library( gpu_framework ${GpuFrameworkSrcTop})

target_include_directories( gpu_framework PUBLIC
   ${GPU_FRAMEWORK_DIR}/include
   ${GPU_FRAMEWORK_DIR}/include/utils
   ${GPU_FRAMEWORK_DIR}/include/detail
   $ENV{CPU_FRAMEWORK_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include/detail
   $ENV{INPUT_FRAMEWORK_DIR}/include
)

target_link_libraries(gpu_framework MPI::MPI_CXX cublas cusparse) # FIXME: Add CUDA here

set(NVCC_FLAGS_COMMON --expt-relaxed-constexpr)
set(NVCC_FLAGS_RELEASE -O3)
set(NVCC_FLAGS_DEBUG -O0 -G -g)
set(NVCC_FLAGS_PROF -O3 -lineinfo)

target_compile_options(gpu_framework PRIVATE
    # CUDA specific flags
    $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_FLAGS_COMMON} ${NVCC_FLAGS_${CMAKE_BUILD_TYPE}} -Xcompiler ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}>
)

###### Add target and dependencies for unit tests ##############
set( BUILD_TEST $ENV{BUILD_XDF_TESTS})
if (BUILD_TEST)

   set( GPU_FRAMEWORK_TEST_DIR ${GPU_FRAMEWORK_DIR}/test  )
   aux_source_directory( ${GPU_FRAMEWORK_TEST_DIR}/src GpuFrameworkTestSrcTop)

   if(ENABLE_GPU)
      set_source_files_properties(${GpuFrameworkTestSrcTop} PROPERTIES LANGUAGE CUDA)
   endif()

   add_executable( gpu_framework_test ${GpuFrameworkTestSrcTop} )

   target_include_directories( gpu_framework_test PUBLIC
      ${GPU_FRAMEWORK_DIR}/test/include)

   target_link_libraries( gpu_framework_test
      cpu_framework
      gpu_framework
   )

target_compile_options(gpu_framework_test PRIVATE
       # CUDA specific flags
       $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_FLAGS_COMMON} ${NVCC_FLAGS_${CMAKE_BUILD_TYPE}} -Xcompiler ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}>
   )

endif()
