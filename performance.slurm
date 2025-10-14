#!/bin/bash
#SBATCH --job-name=conv_perf_quick
#SBATCH --partition=cits3402
#SBATCH --time=01:00:00
#SBATCH --output=performance_%j.out
#SBATCH --error=performance_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G

# Set OpenMP environment variables for maximum performance
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_NESTED=false
export OMP_WAIT_POLICY=active

# Compile the C program with aggressive optimization
gcc -fopenmp -O3 -march=native -mtune=native -funroll-loops -ffast-math 3.c -o conv_test -lm

echo "# OpenMP 2D Convolution Performance Analysis"
echo "# Generated on $(date)"
echo "# SLURM Job ID: $SLURM_JOB_ID"
echo "# Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# CSV Header - matching Python script expectations
echo "Test_Category,Matrix_H,Matrix_W,Kernel_H,Kernel_W,Threads,Sequential_Time,Basic_Static_Time,Dynamic_Time,Guided_Time,Collapse_Time,Best_Algorithm,Best_Time,Speedup"

# Function to extract detailed timing data
run_detailed_test() {
    local category="$1"
    local matrix_h="$2"
    local matrix_w="$3"
    local kernel_h="$4"
    local kernel_w="$5"
    local threads="$6"
    
    # CRITICAL: Set thread count BEFORE running the program
    export OMP_NUM_THREADS=$threads
    
    # Run the actual test
    output=$(./conv_test -H $matrix_h -W $matrix_w -kH $kernel_h -kW $kernel_w 2>/dev/null)
    
    # Check if output is valid
    if [ -z "$output" ]; then
        return 1
    fi
    
    # Extract all timing information
    seq_time=$(echo "$output" | grep "Sequential convolution time:" | sed 's/Sequential convolution time: \([0-9.]*\) seconds/\1/')
    basic_time=$(echo "$output" | grep "Parallel basic static time:" | sed 's/Parallel basic static time: \([0-9.]*\) seconds.*/\1/')
    dynamic_time=$(echo "$output" | grep "Parallel dynamic scheduling time:" | sed 's/Parallel dynamic scheduling time: \([0-9.]*\) seconds.*/\1/')
    guided_time=$(echo "$output" | grep "Parallel guided scheduling time:" | sed 's/Parallel guided scheduling time: \([0-9.]*\) seconds.*/\1/')
    collapse_time=$(echo "$output" | grep "Parallel collapse approach time:" | sed 's/Parallel collapse approach time: \([0-9.]*\) seconds.*/\1/')
    
    # Extract best algorithm info
    best_algo=$(echo "$output" | grep "Fastest parallel implementation:" | sed 's/Fastest parallel implementation: \([^(]*\) (.*/\1/' | tr -d ' ')
    best_time=$(echo "$output" | grep "Fastest parallel implementation:" | sed 's/.*(\([0-9.]*\) seconds.*/\1/')
    speedup=$(echo "$output" | grep "Fastest parallel implementation:" | sed 's/.*seconds, \([0-9.]*\)x speedup).*/\1/')
    
    # Handle missing values
    [ -z "$seq_time" ] && seq_time="0.000"
    [ -z "$basic_time" ] && basic_time="0.000"
    [ -z "$dynamic_time" ] && dynamic_time="0.000"
    [ -z "$guided_time" ] && guided_time="0.000"
    [ -z "$collapse_time" ] && collapse_time="0.000"
    [ -z "$best_time" ] && best_time="0.000"
    [ -z "$best_algo" ] && best_algo="Unknown"
    [ -z "$speedup" ] && speedup="0.00"
    
    echo "$category,$matrix_h,$matrix_w,$kernel_h,$kernel_w,$threads,$seq_time,$basic_time,$dynamic_time,$guided_time,$collapse_time,$best_algo,$best_time,$speedup"
}

echo ""
echo "# === 1. ALGORITHM COMPARISON ==="
echo "# Testing different OpenMP scheduling approaches"
for matrix_size in 200 400 600 800 1000 1500 2000 5000 10000; do
    for kernel_size in 3 5 7 11 15 21 31; do
        for threads in 1 4 8 16 32 48 96; do
            run_detailed_test "Algorithm_Comparison" $matrix_size $matrix_size $kernel_size $kernel_size $threads
        done
    done
done

echo ""
echo "# Performance analysis complete!"
