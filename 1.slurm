#!/bin/bash
#SBATCH --job-name=a1_updated
#SBATCH --partition=cits3402
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err

# Match OpenMP threads to SLURM allocation and pin to physical cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Optional debug:
# export OMP_DISPLAY_ENV=VERBOSE

echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS      = $OMP_NUM_THREADS"
echo "OMP_PLACES           = $OMP_PLACES"
echo "OMP_PROC_BIND        = $OMP_PROC_BIND"
echo

# Compile
gcc -fopenmp -O3 -march=native 1.c -o a1

# Run with files (sequential + parallel outputs)
srun --hint=nomultithread ./a1 -H 1000 -W 1000 -k 3