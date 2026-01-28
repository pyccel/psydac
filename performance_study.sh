#!/bin/bash

#SBATCH --job-name=scaling_study

#SBATCH --output=log_%a.out

#SBATCH --partition=p.tok.openmp          #Queue/Partition

#SBATCH --qos=p.tok.openmp.2h            #Quality of Service (see below): p.tok.2h, p.tok.48, tok.debug

#SBATCH --nodes=1                 # Request 1 physical node

#SBATCH --ntasks-per-node=32        # Ensure the node can handle up to 32 tasks

#SBATCH --cpus-per-task=1           # 1 CPU per MPI process for a scaling study

#SBATCH --mem 185GB                #Set mem./node requirement (default: 63000 MB, max: 185GB)

#SBATCH --time=02:00:00           # Set a reasonable time limit

#SBATCH --array=1,4,8,16,32     # The number of processes to test

##



# SLURM_ARRAY_TASK_ID will correspond to 4, 8, 16, or 32

echo "Running with $SLURM_ARRAY_TASK_ID processes"

cat /proc/meminfo

echo loading modules ...

module purge

module load gcc/14

module load openmpi/5.0

module load hdf5-mpi/1.14.1



source ~/virtual_env/vpsydac2/bin/activate

export LD_LIBRARY_PATH=$HDF5_HOME/lib:$GCC_HOME/lib:$OPENMPI_HOME/lib




# Standard MPI/OpenMP env variables

export OMP_NUM_THREADS=1 

export OMP_PLACES=cores

export OMP_PROC_BIND=close



# Use mpirun or srun depending on how your main.py is parallelized

# If using multiprocessing/concurrent.futures, just call python directly

srun -n $SLURM_ARRAY_TASK_ID python performance_topetsc.py >> "results_${SLURM_ARRAY_TASK_ID}.out"
