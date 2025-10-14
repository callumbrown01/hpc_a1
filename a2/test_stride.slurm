#!/bin/bash
#SBATCH --account=courses0101
#SBATCH --partition=work
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=stride_impact_test
#SBATCH --output=stride_impact_%j.out
#SBATCH --error=stride_impact_%j.err
#SBATCH --exclusive

# Load required modules
module load cray-mpich

echo "======================================="
echo "Stride Impact Performance Testing"
echo "======================================="

# Compile the program
echo "Compiling a2_performance.c..."
cc -fopenmp -o a2_performance a2_performance.c
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

# Test configurations: Matrix size, Kernel size, Stride H, Stride W
test_configs=(
    "1000 3 1 1"     # Small: 100x100 with 3x3 kernel =100x100
    "1000 3 2 1"     # Medium: 500x500 with 5x5 kernel  
    "1000 3 1 2"    # Large: 1000x1000 with 3x3 kernel
    "1000 3 2 2"    
    "5000 3 1 1"
    "5000 3 1 3"
    "5000 3 3 1"
    "5000 3 3 3"
)

# Number of threads/processes to test
omp_threads=(1 2 4 8 16 32)
mpi_procs=(1 2 4 8 16 32)

echo ""
echo "Test Matrix Configurations:"
for i in "${!test_configs[@]}"; do
    read -r H kH sH sW <<< "${test_configs[$i]}"
    echo "  Config $i: ${H}x${H} input, ${kH}x${kH} kernel, stride ${sH}x${sW}"
done
echo ""

# Function to run a single test
run_test() {
    local mode=$1
    local H=$2
    local kH=$3
    local sH=$4
    local sW=$5
    local procs=$6
    local threads=$7
    
    export OMP_NUM_THREADS=$threads
    
    if [ "$mode" = "seq" ]; then
        echo -n "  Sequential: "
        srun -n 1 ./a2_performance seq $H $kH $sH $sW 2>/dev/null | grep "Time:"
    elif [ "$mode" = "omp" ]; then
        echo -n "  OpenMP (${threads}t): "
        srun -n 1 ./a2_performance omp $H $kH $sH $sW 2>/dev/null | grep "Time:"
    elif [ "$mode" = "mpi" ]; then
        echo -n "  MPI (${procs}p): "
        srun -n $procs ./a2_performance mpi $H $kH $sH $sW 2>/dev/null | grep "Time:"
    elif [ "$mode" = "hybrid" ]; then
        echo -n "  Hybrid (${procs}p×${threads}t): "
        srun -n $procs ./a2_performance hybrid $H $kH $sH $sW 2>/dev/null | grep "Time:"
    fi
}


# Prepare CSV output
CSV_OUT="results/stride_impact.csv"
mkdir -p results
echo "config_id,matrix_size,kernel_size,stride_h,stride_w,mode,threads,procs,time_seconds,speedup" > "$CSV_OUT"

# Run tests for each configuration
for i in "${!test_configs[@]}"; do
    read -r H kH sH sW <<< "${test_configs[$i]}"
    echo "======================================="
    echo "Test Config $i: ${H}x${H} input, ${kH}x${kH} kernel, stride ${sH}x${sW}"
    echo "======================================="

    # Sequential baseline (always run with 1 thread)
    SEQ_TIME=$(srun -n 1 ./a2_performance seq $H $kH $sH $sW 2>/dev/null | grep "Time:" | awk '{print $4}')
    echo "  Sequential: $SEQ_TIME seconds"
    echo "$i,$H,$kH,$sH,$sW,seq,1,1,$SEQ_TIME,1.0" >> "$CSV_OUT"

    # OpenMP scaling - test different thread counts
    echo "OpenMP Scaling:"
    for threads in "${omp_threads[@]}"; do
        if [ $threads -le 32 ]; then
            OMP_TIME=$(srun -n 1 ./a2_performance omp $H $kH $sH $sW 2>/dev/null | grep "Time:" | awk '{print $4}')
            SPEEDUP=$(echo "scale=4; $SEQ_TIME / $OMP_TIME" | bc)
            echo "$i,$H,$kH,$sH,$sW,omp,$threads,1,$OMP_TIME,$SPEEDUP" >> "$CSV_OUT"
            echo "  OpenMP ($threads t): $OMP_TIME seconds, speedup: $SPEEDUP"
        fi
    done

    # MPI scaling - test different process counts
    echo "MPI Scaling:"
    for procs in "${mpi_procs[@]}"; do
        if [ $procs -le 32 ]; then
            MPI_TIME=$(srun -n $procs ./a2_performance mpi $H $kH $sH $sW 2>/dev/null | grep "Time:" | awk '{print $4}')
            SPEEDUP=$(echo "scale=4; $SEQ_TIME / $MPI_TIME" | bc)
            echo "$i,$H,$kH,$sH,$sW,mpi,1,$procs,$MPI_TIME,$SPEEDUP" >> "$CSV_OUT"
            echo "  MPI ($procs p): $MPI_TIME seconds, speedup: $SPEEDUP"
        fi
    done

    # Hybrid - test a few good combinations
    echo "Hybrid Scaling:"
    for combo in "2 4" "4 2" "4 4" "8 4" "8 8" "16 8" "16 16"; do
        set -- $combo; procs=$1; threads=$2
        HYB_TIME=$(srun -n $procs ./a2_performance hybrid $H $kH $sH $sW 2>/dev/null | grep "Time:" | awk '{print $4}')
        SPEEDUP=$(echo "scale=4; $SEQ_TIME / $HYB_TIME" | bc)
        echo "$i,$H,$kH,$sH,$sW,hybrid,$threads,$procs,$HYB_TIME,$SPEEDUP" >> "$CSV_OUT"
        echo "  Hybrid ($procs p x $threads t): $HYB_TIME seconds, speedup: $SPEEDUP"
    done
done

echo ""
echo "======================================="
echo "Stride Impact Testing Complete"
echo "======================================="