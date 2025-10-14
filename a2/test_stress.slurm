#!/bin/bash --login
#SBATCH --account=courses0101
#SBATCH --partition=work
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=max_capability_test
#SBATCH --output=max_capability_%j.out
#SBATCH --error=max_capability_%j.err
#SBATCH --exclusive

# Load required modules
module load cray-mpich

echo "======================================="
echo "Maximum Capability Stress Test"
echo "Using 4 nodes × 64 MPI processes × 8 OpenMP threads = 512 total cores"
echo "Memory limit: ~256 GB per node, 1 TB total"
echo "Time limit: 15 minutes per test"
echo "======================================="

# Set MPI environment variables for Setonix
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu4 < /dev/urandom)

# Compile the program
echo "Compiling a2_performance.c..."
cc -fopenmp -o a2_performance a2_performance.c
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

# Progressive stress test configurations: Matrix size, Kernel size, Stride H, Stride W
# Start small and progressively increase until we hit memory or time limits
test_configs=(
    "40000 101 1 1"     # ~25 GB per matrix
    "40000 121 1 1"     # ~25 GB per matrix
    "40000 141 1 1"     # ~25 GB per matrix
    "40000 161 1 1"     # ~25 GB per matrix
    "40000 181 1 1"     # ~25 GB per matrix
    "40000 201 1 1"     # ~25 GB per matrix
)

# Maximum configuration: 64 MPI processes × 8 OpenMP threads = 512 cores
MAX_PROCS=64
MAX_THREADS=8

echo ""
echo "Test Matrix Configurations (estimated memory per matrix):"
for i in "${!test_configs[@]}"; do
    read -r H kH sH sW <<< "${test_configs[$i]}"
    MEM_GB=$(echo "scale=1; $H * $H * 8 / 1024 / 1024 / 1024" | bc)
    echo "  Config $i: ${H}x${H} input, ${kH}x${kH} kernel, stride ${sH}x${sW} (~${MEM_GB} GB)"
done
echo ""

# Prepare CSV output for maximum capability results
CSV_OUT="results/max_capability.csv"
mkdir -p results
echo "matrix_size,kernel_size,memory_gb,time_seconds,status,cores_used" > "$CSV_OUT"

echo "======================================="
echo "Starting Progressive Stress Testing"
echo "Using Hybrid MPI+OpenMP: $MAX_PROCS processes × $MAX_THREADS threads"
echo "======================================="

# Run progressive stress tests
export OMP_NUM_THREADS=$MAX_THREADS

for i in "${!test_configs[@]}"; do
    read -r H kH sH sW <<< "${test_configs[$i]}"
    
    # Calculate expected memory usage
    MEM_GB=$(echo "scale=1; $H * $H * 8 / 1024 / 1024 / 1024" | bc)
    CORES_USED=$((MAX_PROCS * MAX_THREADS))
    
    echo ""
    echo "======================================="
    echo "Testing ${H}x${H} matrix (estimated ${MEM_GB} GB)"
    echo "Using $MAX_PROCS MPI processes × $MAX_THREADS OpenMP threads = $CORES_USED cores"
    echo "======================================="
    
    # Set timeout for this test (13 minutes max, leaving 2 min buffer)
    timeout 780s bash -c "
        start_time=\$(date +%s)
        echo 'Starting hybrid computation...'
        
        # Run the maximum capability test
        result=\$(srun -n $MAX_PROCS --cpus-per-task=$MAX_THREADS ./a2_performance hybrid $H $kH $sH $sW 2>&1)
        exit_code=\$?
        
        end_time=\$(date +%s)
        elapsed=\$((end_time - start_time))
        
        if [ \$exit_code -eq 0 ]; then
            time_result=\$(echo \"\$result\" | grep 'Time:' | awk '{print \$4}')
            if [ -n \"\$time_result\" ]; then
                echo \"SUCCESS: Completed in \$time_result seconds\"
                echo \"$H,$kH,$MEM_GB,\$time_result,SUCCESS,$CORES_USED\" >> \"$CSV_OUT\"
            else
                echo \"ERROR: No time result found\"
                echo \"$H,$kH,$MEM_GB,\$elapsed,ERROR,$CORES_USED\" >> \"$CSV_OUT\"
            fi
        else
            echo \"FAILED: Exit code \$exit_code after \$elapsed seconds\"
            echo \"$H,$kH,$MEM_GB,\$elapsed,FAILED,$CORES_USED\" >> \"$CSV_OUT\"
        fi
    "
    
    test_exit=$?
    if [ $test_exit -eq 124 ]; then
        echo "TIMEOUT: Test exceeded 13 minutes"
        echo "$H,$kH,$MEM_GB,780,TIMEOUT,$CORES_USED" >> "$CSV_OUT"
        echo ""
        echo "======================================="
        echo "MAXIMUM CAPABILITY REACHED!"
        echo "Last successful size was from previous test"
        echo "Matrix size $H x $H exceeded time limit"
        echo "======================================="
        break
    elif [ $test_exit -ne 0 ]; then
        echo "MEMORY LIMIT: Test failed, likely out of memory"
        echo "$H,$kH,$MEM_GB,0,MEMORY_LIMIT,$CORES_USED" >> "$CSV_OUT"
        echo ""
        echo "======================================="
        echo "MEMORY LIMIT REACHED!"
        echo "Matrix size $H x $H exceeded memory capacity"
        echo "======================================="
        break
    fi
    
    echo "Test completed successfully, proceeding to next size..."
done

echo ""
echo "======================================="
echo "Maximum Capability Testing Complete!"
echo "Results saved to: $CSV_OUT"
echo "======================================="
