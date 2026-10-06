module load cmake/3.31.6
module load prgenv/expert
module load nvidia/25.9
module load hpcx-openmpi/2.21.3-cuda:nvidia:25.9

export FC=nvfortran

# Record the RPATH in the executable
export LD_RUN_PATH=$LD_LIBRARY_PATH
