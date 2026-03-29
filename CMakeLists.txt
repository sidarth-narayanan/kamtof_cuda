#CMake for the Laplace solver

cmake_minimum_required( VERSION 3.20 )

set( CMAKE_CXX_STANDARD 23 )

project( "Laplace solver using KAMTOF" LANGUAGES CXX VERSION 0.0.1)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Pass the Device flag to the solver source code
if(NOT DEFINED ENABLE_GPU) # By default build the ENABLE GPU ON VERSION
   set(ENABLE_GPU ON)
endif()

find_package (MPI REQUIRED)

# Build in release mode by default
if( CMAKE_BUILD_TYPE STREQUAL "" )
   set( CMAKE_BUILD_TYPE RELEASE )
else()
   string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

set( CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG -Werror" )
set( CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fno-inline-functions -Wunused-result" )
set( CMAKE_CXX_FLAGS_ASAN    "-g -O3 -DNDEBUG -fsanitize=address -fno-omit-frame-pointer " )
set( CMAKE_CXX_FLAGS_RELWITHDEBINFO    "-g -O3 -DNDEBUG " )

if(ENABLE_GPU)
   set(NVCC_FLAGS_COMMON --expt-relaxed-constexpr)
   set(NVCC_FLAGS_RELEASE -O3)
   set(NVCC_FLAGS_DEBUG -O0 -G -g)
   set(NVCC_FLAGS_PROF -O3 -lineinfo)

   if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86)
   endif()

   set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
   set(CMAKE_CUDA_HOST_COMPILER ${CXX})
   add_definitions(-DENABLE_GPU)
   enable_language(CUDA)
   set(CMAKE_CUDA_ARCHITECTURES 86)
   set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
   message(STATUS "Buidling for the following CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
   find_package(CUDAToolkit REQUIRED)
endif()

# Build the XDF tests by default
if(NOT DEFINED BUILD_XDF_TESTS)
   set( ENV{BUILD_XDF_TESTS} ON)
   message( STATUS "Building XDF tests by default. To turn it OFF, run CMake with -DBUILD_XDF_TESTS=OFF" )
else()
   set( ENV{BUILD_XDF_TESTS} "${BUILD_XDF_TESTS}")
   if(BUILD_XDF_TESTS)
      message(STATUS "Building XDF tests. To turn it OFF, run CMake with -DBUILD_XDF_TESTS=OFF")
   else()
      message(STATUS "NOT Building XDF tests. To turn it ON, run CMake with -DBUILD_XDF_TESTS=ON")
   endif()
endif()


if(ENABLE_GPU)
   if(GPU_MEM_LOG)
      add_definitions(-DGPU_MEM_LOG)
   endif()
endif()

set( ENV{PROJECT_ROOT_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set( ENV{INPUT_FRAMEWORK_DIR} $ENV{PROJECT_ROOT_DIR}/input )
set( ENV{CPU_FRAMEWORK_DIR} $ENV{PROJECT_ROOT_DIR}/cpu_framework )
set( ENV{CPU_SOLVER_DIR} $ENV{PROJECT_ROOT_DIR}/solver/cpu )

# Add the framework subdirs
add_subdirectory( $ENV{INPUT_FRAMEWORK_DIR} )

add_subdirectory( $ENV{CPU_FRAMEWORK_DIR} )

if (ENABLE_GPU)
   set( ENV{GPU_FRAMEWORK_DIR} $ENV{PROJECT_ROOT_DIR}/gpu_framework )
   add_subdirectory( $ENV{GPU_FRAMEWORK_DIR} )
endif()

# Add solver source
aux_source_directory( $ENV{CPU_SOLVER_DIR}/src CpuSolverSrcTop)

if(ENABLE_GPU)
   set( ENV{GPU_SOLVER_DIR} $ENV{PROJECT_ROOT_DIR}/solver/gpu )
   aux_source_directory( $ENV{GPU_SOLVER_DIR}/src GpuSolverSrcTop)
   set_source_files_properties(${CpuSolverSrcTop} PROPERTIES LANGUAGE CUDA)
   set_source_files_properties(${GpuSolverSrcTop} PROPERTIES LANGUAGE CUDA)
   add_executable(solver ${CpuSolverSrcTop} ${GpuSolverSrcTop})
else()
   add_executable(solver ${CpuSolverSrcTop})
endif()

target_compile_options(solver PRIVATE
   # CUDA specific flags
   $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_FLAGS_COMMON} ${NVCC_FLAGS_${CMAKE_BUILD_TYPE}} -Xcompiler ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}>
)

target_include_directories( solver PUBLIC
   $ENV{CPU_SOLVER_DIR}/include
   $ENV{INPUT_FRAMEWORK_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include
   $ENV{CPU_FRAMEWORK_DIR}/include/detail
   $ENV{INPUT_FRAMEWORK_DIR}/include
)

if (ENABLE_GPU)
   target_include_directories(solver PUBLIC
      $ENV{GPU_SOLVER_DIR}/include
      $ENV{GPU_FRAMEWORK_DIR}/include
      $ENV{GPU_FRAMEWORK_DIR}/include/detail
      $ENV{GPU_FRAMEWORK_DIR}/include/util)
   target_link_libraries(solver cpu_framework gpu_framework MPI::MPI_CXX)
else()
   target_link_libraries(solver cpu_framework MPI::MPI_CXX)
endif()
