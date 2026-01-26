#!/bin/bash
#SBATCH --job-name=scaling_study
#SBATCH --output=log_%a.out
#SBATCH --nodes=1                 # Request 1 physical node
#SBATCH --exclusive               # Reserve the entire node for yourself
#SBATCH --time=10:00:00           # Set a reasonable time limit
#SBATCH --array=1,2,4,8,16,32     # The number of processes to test

# SLURM_ARRAY_TASK_ID will correspond to 1, 2, 4, 8, 16, or 32
echo "Running with $SLURM_ARRAY_TASK_ID processes"

echo loading modules ...
module purge
module load gcc/14
module load openmpi/5.0
module load hdf5-mpi/1.14.1

source ~/virtual_env/vpsydac1/bin/activate
export LD_LIBRARY_PATH=$HDF5_HOME/lib:$GCC_HOME/lib:$OPENMPI_HOME/lib


echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting Python job at: $(date)"

# Use mpirun or srun depending on how your main.py is parallelized
# If using multiprocessing/concurrent.futures, just call python directly
srun -n $SLURM_ARRAY_TASK_ID python performance_topetsc.py

echo "Job finished at: $(date)"
