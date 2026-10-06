cmake_minimum_required(VERSION 3.25)

project(gpu_mpi_pingpong LANGUAGES Fortran)

find_package(MPI REQUIRED COMPONENTS Fortran)
find_package(OpenACC REQUIRED COMPONENTS Fortran)

if(NOT MPI_Fortran_HAVE_F08_MODULE)
  message(FATAL_ERROR "The MPI library does not provide the mpi_f08 Fortran module")
endif()

add_executable(gpu_pingpong gpu_pingpong.F90)
target_link_libraries(gpu_pingpong PRIVATE MPI::MPI_Fortran OpenACC::OpenACC_Fortran)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  # Force true device allocations for OpenACC data. On Grace Hopper the
  # compiler could otherwise place buffers in unified/host memory, which would
  # hide the GPU-direct MPI problem this program is meant to reproduce.
  set(GPU_FLAGS -acc=gpu -gpu=mem:separate -Minfo=accel)
  target_compile_options(gpu_pingpong PRIVATE ${GPU_FLAGS})
  target_link_options(gpu_pingpong PRIVATE ${GPU_FLAGS})
endif()
