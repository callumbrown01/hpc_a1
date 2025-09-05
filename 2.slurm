#!/bin/bash
#SBATCH --job-name=performance_analysis
#SBATCH --partition=cits3402
#SBATCH --time=02:00:00
#SBATCH --output=performance_%j.out
#SBATCH --error=error_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=16G

# Set OpenMP environment variables for maximum performance
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_NESTED=false

# Compile the C program with aggressive optimization
gcc -fopenmp -O3 -march=native -mtune=native -funroll-loops -ffast-math 2.c -o 2 -lm

echo "# Performance Analysis Results - CSV Format for Graphing"
echo "# Generated on $(date)"
echo "# SLURM Job ID: $SLURM_JOB_ID"
echo "# Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# CSV Header
echo "Test_Type,Problem_Description,Matrix_Size_H,Matrix_Size_W,Kernel_Size,Threads,Sequential_Time,Parallel_Time,Speedup"

# Function to extract timing data and format as CSV
run_test() {
    local test_type="$1"
    local description="$2"
    local args="$3"
    local threads="$4"
    
    export OMP_NUM_THREADS=$threads
    output=$(./2 $args 2>/dev/null)
    
    # Extract problem size information
    matrix_info=$(echo "$output" | grep "Problem size:")
    if [[ $matrix_info =~ Problem\ size:\ ([0-9]+)x([0-9]+)\ matrix\ with\ ([0-9]+)x([0-9]+)\ kernel ]]; then
        matrix_h="${BASH_REMATCH[1]}"
        matrix_w="${BASH_REMATCH[2]}"
        kernel_size="${BASH_REMATCH[3]}"
    else
        matrix_h="N/A"
        matrix_w="N/A" 
        kernel_size="N/A"
    fi
    
    # Extract timing information with more precise regex
    seq_time=$(echo "$output" | grep "Sequential convolution time:" | sed 's/Sequential convolution time: \([0-9.]*\) seconds/\1/')
    par_time=$(echo "$output" | grep "Parallel convolution time:" | sed 's/Parallel convolution time: \([0-9.]*\) seconds/\1/')
    speedup=$(echo "$output" | grep "Speedup:" | sed 's/Speedup: \([0-9.]*\)x/\1/')
    
    # Handle missing values
    if [ -z "$seq_time" ]; then seq_time="0.000"; fi
    if [ -z "$par_time" ]; then par_time="0.000"; fi
    if [ -z "$speedup" ]; then speedup="0.00"; fi
    
    echo "$test_type,$description,$matrix_h,$matrix_w,$kernel_size,$threads,$seq_time,$par_time,$speedup"
}

echo ""
echo "# === INPUT FILE TESTS ==="
run_test "InputFile" "f0_g0" "-f f0.txt -g g0.txt" 1
run_test "InputFile" "f0_g0" "-f f0.txt -g g0.txt" 8
run_test "InputFile" "f0_g0" "-f f0.txt -g g0.txt" 16
run_test "InputFile" "f1_g1" "-f f1.txt -g g1.txt" 1
run_test "InputFile" "f1_g1" "-f f1.txt -g g1.txt" 8
run_test "InputFile" "f1_g1" "-f f1.txt -g g1.txt" 16
run_test "InputFile" "f2_g2" "-f f2.txt -g g2.txt" 1
run_test "InputFile" "f2_g2" "-f f2.txt -g g2.txt" 8
run_test "InputFile" "f2_g2" "-f f2.txt -g g2.txt" 16
run_test "InputFile" "f3_g3" "-f f3.txt -g g3.txt" 1
run_test "InputFile" "f3_g3" "-f f3.txt -g g3.txt" 8
run_test "InputFile" "f3_g3" "-f f3.txt -g g3.txt" 16

echo ""
echo "# === THREAD SCALING TESTS ==="
# Thread scaling on different problem sizes
for size in 100 500 1000 1500 2000 5000 10000; do
    for threads in 1 2 4 8 16 24 32 48 64 96; do
        run_test "ThreadScaling" "Size_${size}x${size}" "-H $size -W $size -k 5" $threads
    done
done

echo ""
echo "# === MATRIX SIZE SCALING TESTS ==="
# Matrix size scaling with fixed thread count
for size in 100 200 300 400 500 750 1000 1250 1500 2000; do
    run_test "MatrixScaling" "Size_${size}x${size}" "-H $size -W $size -k 5" 8
done

echo ""
echo "# === KERNEL SIZE SCALING TESTS ==="
# Kernel size scaling with fixed matrix and thread count
for kernel in 3 5 7 9 11 13 15; do
    run_test "KernelScaling" "Kernel_${kernel}x${kernel}" "-H 1000 -W 1000 -k $kernel" 8
done

echo ""
echo "# === COMPREHENSIVE PERFORMANCE MATRIX ==="
# Comprehensive test matrix
for matrix_size in 500 1000 1500; do
    for kernel_size in 3 5 7 9; do
        for threads in 1 2 4 8 16 32 48 64 96; do
            run_test "Comprehensive" "M${matrix_size}_K${kernel_size}" "-H $matrix_size -W $matrix_size -k $kernel_size" $threads
        done
    done
done

echo ""
