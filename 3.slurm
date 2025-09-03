#!/bin/bash
#SBATCH --job-name=a1
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G

# Compile the C program
gcc -fopenmp -O3 -march=native 3.c -o 3 -lm

# Run the program
./3 -f f3.txt -g g3.txt 