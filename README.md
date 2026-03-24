# KAMTOF : Kernel-level Access-based Memory Transfer Optimization Framework

KAMTOF is a CUDA based GPU data framework for rapid incremental GPU porting of a large-scale CFD CPU code

## Pre-requisites for KAMTOF

KAMTOF requires the following tools:
- CUDA toolkit v13.0+
- HPCX MPI Library 2.21+

<a name="env_setup"></a>
## Environment Setup for KAMTOF

KAMTOF provides an environment [setup script](scripts/setup_env.sh) in the scripts folder to setup the environment needed for building. The setup script requires the following environment variables to be setup in the [export_root.sh](scripts/export_roots.sh) script:

- CUDA_ROOT
- HPCX_ROOT

Once the values of these ROOTs are edited in the [export_root.sh](scripts/export_roots.sh) script, the environment for building and running can be setup by performing the following command:

```
source ./scripts/setup_env.sh
```

NOTE: Moving folder/files inside the KAMTOF base folder leads to undefined behaviour   


## Building and Running KAMTOF

There are two ways to build and run KAMTOF:
- [Build script](scipts/build.sh) (Recommended)
- Manual compilation

### Build Script

The [build script](scipts/build.sh) can be run to build KAMTOF once the appropriate paths have been set in [export_root.sh](scripts/export_roots.sh).

For most users, simply executing `./scripts/build.sh` will compile KAMTOF in *RELEASE* mode (i.e., -O3 optimizations). For more information on compilation options, please execute `./scripts/build.sh -h`.

The default location for the build directory is one level above the KAMTOF base directory (`../build_release` for *RELEASE* mode and `../build_debug` for *DEBUG* mode). 

[build_kamtof.sh](scripts/build_kamtof.sh) can be used to compile KAMTOF in both *RELEASE* and *DEBUG* modes by invoking `./scripts/build_kamtof.sh`.


### Manual Compilation

For manual compilation, users need to setup the environment as mentioned in the [environment setup section](#env_setup).

After the environment is setup, CMake and make can be run as mentioned below with the following options:

#### CMake options

These are the CMake options provided by KAMTOF. Please preface all these options with -D flag when using in the CMake command.

- CMAKE_BUILD_TYPE : Build type option
  - Options: RELEASE/DEBUG/MAP/ASAN
  - Default: RELEASE 

- ENABLE_GPU : To turn ON the GPU solver
  - Options: ON/OFF
  - Default: OFF
 
- BUILD_XDF_TESTS :  Flag to turn ON the unit-test builds for the CPU Data Framework (CDF) and GPU Data Framework (GDF). Note that currently CDF unit tests are not built when ENABLE_GPU is ON
  - Options: ON/OFF
  - Default: ON
 
Here is an example of a CMake command to build for NVIDIA GPUs (arch 86) in release mode:

```
mkdir build_release
cd build_release
cmake -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_GPU=ON ../
make -j8
```

## Running KAMTOF

In order to run KAMTOF, users must source the [setup script](scripts/setup_env.sh).

KAMTOF contains 3 different executables: 
- `cpu_framework/cpu_framework_test` : Test certain basic functionalities of the CPU framework (Will not be built is ENABLE_GPU is ON)
- `gpu_framework/gpu_framework_test` : Test certain basic functionalities of the GPU framework
- `solver` : The actual Laplace Solver
  - The input files, [laplace.in](input_files/laplace.in), for the solver are placed in the [input_files](input_files) folder
 
## KAMTOF Artifact Description
This artifact description ([KAMTOF_AD.pdf](https://github.com/user-attachments/files/21623194/KAMTOF_AD.pdf)) is intended to demonstrate a multi-CPU, multi-GPU Laplace solver, based on an existing MPI-based CPU Laplace solver ported using KAMTOF. We provide detailed instructions to generate results that shows the speed up of the GPU solver compared to the CPU solver and the accuracy verification of the GPU solver results.

## Contributors
- This project was primarily forked off https://github.com/ConvergentScience/KAMTOF
- Currently this is maintained by Sidarth Narayanan (sidarth.narayanan07@gmail.com)
