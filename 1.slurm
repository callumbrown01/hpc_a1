#!/bin/bash
#SBATCH --job-name=a1
#SBATCH --partition=cits3402
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err

# Compile the C program
gcc -fopenmp 1.c -o 1

# Run the program
./1 -f f0.txt -g g0.txt -o output.txt