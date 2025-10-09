#!/bin/bash --login

#SBATCH --job-name=a2_scaling
#SBATCH --account=courses0101
#SBATCH --partition=work
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err

# Load modules
module load cray-mpich

# Set MPI environment variables for Setonix
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu4 < /dev/urandom)

cd /home/cbrown1 || exit 1
mkdir -p results
cd /home/cbrown1

# Compile
echo "======================================"
echo "Compiling a2_performance.c..."
echo "======================================"
cc -fopenmp -o a2_performance a2_performance.c
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"
echo ""

echo "======================================"
echo "Scaling Performance Study"
echo "======================================"
echo "Testing matrix sizes: 100x100, 500x500, 1000x1000, 10000x10000, 50000x50000"
echo "With 5x5 kernel for all tests"
echo "Date: $(date)"
echo ""

# Problem sizes to test (matrix_size kernel_size)
SIZES=(
    "100 2"
    "500 2" 
    "1000 2"
    "5000 2"
    "10000 2"
    "50000 2"
    "100000 2   
)

# Create master CSV with all results
CSV_MASTER="results/scaling_master.csv"
echo "matrix_size,mode,threads_or_processes,run,time_seconds,speedup,efficiency_percent" > $CSV_MASTER

# Loop through each problem size
for size_info in "${SIZES[@]}"; do
    read -r MATRIX_SIZE KERNEL_SIZE <<< "$size_info"
    
    echo "======================================"
    echo "Testing ${MATRIX_SIZE}x${MATRIX_SIZE} matrix with ${KERNEL_SIZE}x${KERNEL_SIZE} kernel"
    echo "======================================"
    
    # Create size-specific CSV
    CSV_SIZE="results/scaling_${MATRIX_SIZE}x${MATRIX_SIZE}.csv"
    echo "mode,threads_or_processes,run,time_seconds,speedup,efficiency_percent" > $CSV_SIZE
    
    # ===========================
    # 1. SEQUENTIAL BASELINE
    # ===========================
    echo "1. Sequential Baseline..."
    
    TOTAL_SEQ_TIME=0
    for run in 1 2 3; do
        echo "  Run $run/3..."
        TIME=$(srun -n 1 -c 1 ./a2_performance seq $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
        echo "    Time: $TIME seconds"
        echo "seq,1,$run,$TIME,1.0,100.0" >> $CSV_SIZE
        echo "$MATRIX_SIZE,seq,1,$run,$TIME,1.0,100.0" >> $CSV_MASTER
        TOTAL_SEQ_TIME=$(echo "$TOTAL_SEQ_TIME + $TIME" | bc)
    done
    
    # Calculate baseline average
    BASELINE=$(echo "scale=6; $TOTAL_SEQ_TIME / 3" | bc)
    echo "  Average baseline: $BASELINE seconds"
    echo ""
    
    # ===========================
    # 2. OPENMP SCALING
    # ===========================
    echo "2. OpenMP Thread Scaling..."
    
    for threads in 2 4 8 16 32 64; do
        echo "  Testing with $threads threads..."
        export OMP_NUM_THREADS=$threads
        
        TOTAL_OMP_TIME=0
        for run in 1 2 3; do
            TIME=$(srun -n 1 -c $threads ./a2_performance omp $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            TOTAL_OMP_TIME=$(echo "$TOTAL_OMP_TIME + $TIME" | bc)
        done
        
        # Calculate average for this thread count
        AVG_TIME=$(echo "scale=6; $TOTAL_OMP_TIME / 3" | bc)
        SPEEDUP=$(echo "scale=4; $BASELINE / $AVG_TIME" | bc)
        EFFICIENCY=$(echo "scale=2; $SPEEDUP / $threads * 100" | bc)
        
        echo "    Average: $AVG_TIME s (speedup: ${SPEEDUP}x, efficiency: ${EFFICIENCY}%)"
        
        # Record individual runs
        for run in 1 2 3; do
            TIME=$(srun -n 1 -c $threads ./a2_performance omp $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            RUN_SPEEDUP=$(echo "scale=4; $BASELINE / $TIME" | bc)
            RUN_EFFICIENCY=$(echo "scale=2; $RUN_SPEEDUP / $threads * 100" | bc)
            echo "omp,$threads,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_SIZE
            echo "$MATRIX_SIZE,omp,$threads,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_MASTER
        done
    done
    echo ""
    
    # ===========================
    # 3. MPI SCALING
    # ===========================
    echo "3. MPI Process Scaling..."
    
    for processes in 2 4 8 16 32 64; do
        echo "  Testing with $processes processes..."
        
        TOTAL_MPI_TIME=0
        for run in 1 2 3; do
            TIME=$(srun -n $processes -c 1 ./a2_performance mpi $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            TOTAL_MPI_TIME=$(echo "$TOTAL_MPI_TIME + $TIME" | bc)
        done
        
        # Calculate average for this process count
        AVG_TIME=$(echo "scale=6; $TOTAL_MPI_TIME / 3" | bc)
        SPEEDUP=$(echo "scale=4; $BASELINE / $AVG_TIME" | bc)
        EFFICIENCY=$(echo "scale=2; $SPEEDUP / $processes * 100" | bc)
        
        echo "    Average: $AVG_TIME s (speedup: ${SPEEDUP}x, efficiency: ${EFFICIENCY}%)"
        
        # Record individual runs
        for run in 1 2 3; do
            TIME=$(srun -n $processes -c 1 ./a2_performance mpi $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            RUN_SPEEDUP=$(echo "scale=4; $BASELINE / $TIME" | bc)
            RUN_EFFICIENCY=$(echo "scale=2; $RUN_SPEEDUP / $processes * 100" | bc)
            echo "mpi,$processes,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_SIZE
            echo "$MATRIX_SIZE,mpi,$processes,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_MASTER
        done
    done
    echo ""
    
    # ===========================
    # 4. HYBRID MPI+OPENMP
    # ===========================
    echo "4. Hybrid MPI+OpenMP Scaling..."
    
    # Test various combinations: processes × threads = total cores
    HYBRID_CONFIGS=(
        "2 2"    # 2 processes × 2 threads = 4 cores
        "4 2"    # 4 processes × 2 threads = 8 cores  
        "8 2"    # 8 processes × 2 threads = 16 cores
        "16 2"   # 16 processes × 2 threads = 32 cores
        "32 2"   # 32 processes × 2 threads = 64 cores
        "2 4"    # 2 processes × 4 threads = 8 cores
        "4 4"    # 4 processes × 4 threads = 16 cores
        "8 4"    # 8 processes × 4 threads = 32 cores
        "16 4"   # 16 processes × 4 threads = 64 cores
    )
    
    for config in "${HYBRID_CONFIGS[@]}"; do
        read -r PROCESSES THREADS <<< "$config"
        TOTAL_CORES=$((PROCESSES * THREADS))
        
        echo "  Testing with $PROCESSES processes × $THREADS threads = $TOTAL_CORES cores..."
        export OMP_NUM_THREADS=$THREADS
        
        TOTAL_HYBRID_TIME=0
        for run in 1 2 3; do
            TIME=$(srun -n $PROCESSES -c $THREADS ./a2_performance hybrid $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            TOTAL_HYBRID_TIME=$(echo "$TOTAL_HYBRID_TIME + $TIME" | bc)
        done
        
        # Calculate average for this configuration
        AVG_TIME=$(echo "scale=6; $TOTAL_HYBRID_TIME / 3" | bc)
        SPEEDUP=$(echo "scale=4; $BASELINE / $AVG_TIME" | bc)
        EFFICIENCY=$(echo "scale=2; $SPEEDUP / $TOTAL_CORES * 100" | bc)
        
        echo "    Average: $AVG_TIME s (speedup: ${SPEEDUP}x, efficiency: ${EFFICIENCY}%)"
        
        # Record individual runs
        for run in 1 2 3; do
            TIME=$(srun -n $PROCESSES -c $THREADS ./a2_performance hybrid $MATRIX_SIZE $KERNEL_SIZE 1 1 2>&1 | grep "Time:" | awk '{print $4}')
            RUN_SPEEDUP=$(echo "scale=4; $BASELINE / $TIME" | bc)
            RUN_EFFICIENCY=$(echo "scale=2; $RUN_SPEEDUP / $TOTAL_CORES * 100" | bc)
            echo "hybrid,$TOTAL_CORES,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_SIZE
            echo "$MATRIX_SIZE,hybrid,$TOTAL_CORES,$run,$TIME,$RUN_SPEEDUP,$RUN_EFFICIENCY" >> $CSV_MASTER
        done
    done
    echo ""
    
    echo "Completed ${MATRIX_SIZE}x${MATRIX_SIZE} matrix testing"
    echo "Results saved to: $CSV_SIZE"
    echo ""
done

echo "======================================"
echo "All scaling tests completed!"
echo "Master results saved to: $CSV_MASTER"
echo "Individual size results in: results/scaling_*x*.csv"
echo "======================================"