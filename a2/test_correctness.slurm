#!/bin/bash --login

#SBATCH --job-name=a2_test
#SBATCH --account=courses0101
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=test_correctness_%j.out
#SBATCH --error=test_correctness_%j.err

# Load modules
module load cray-mpich

# Set MPI environment variables for Setonix
export MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu4 < /dev/urandom)

cd /home/cbrown1 || exit 1

# Compile
echo "Compiling a2_performance.c..."
cc -fopenmp -o a2_performance a2_performance.c
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"
echo ""

echo "======================================"
echo "Testing All Implementations"
echo "======================================"
echo ""

# Function to compare outputs
compare_output() {
    local expected=$1
    local actual=$2
    local tolerance=0.01  # Increased tolerance for floating point
    
    if [ ! -f "$actual" ]; then
        echo "FAIL: Output file not created"
        return 1
    fi
    
    if [ ! -f "$expected" ]; then
        echo "FAIL: Expected file not found"
        return 1
    fi
    
    # Compare using a simple Python script for more robust comparison
    python3 << EOF
import sys

def compare_files(file1, file2, tol):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        
        # Compare dimensions (first line)
        dims1 = lines1[0].strip().split()
        dims2 = lines2[0].strip().split()
        
        if dims1 != dims2:
            print(f"FAIL: Dimension mismatch ({dims1[0]}x{dims1[1]} vs {dims2[0]}x{dims2[1]})")
            return False
        
        # Compare values
        for i, (line1, line2) in enumerate(zip(lines1[1:], lines2[1:]), 1):
            vals1 = [float(x) for x in line1.strip().split()]
            vals2 = [float(x) for x in line2.strip().split()]
            
            if len(vals1) != len(vals2):
                print(f"FAIL: Row {i} has different number of values")
                return False
            
            for j, (v1, v2) in enumerate(zip(vals1, vals2)):
                if abs(v1 - v2) > tol:
                    print(f"FAIL: Value at [{i},{j}] differs: {v1:.6f} vs {v2:.6f}")
                    return False
        
        print("PASS")
        return True

try:
    result = compare_files("$expected", "$actual", $tolerance)
    sys.exit(0 if result else 1)
except Exception as e:
    print(f"FAIL: Comparison error - {e}")
    sys.exit(1)
EOF
    
    return $?
}

total_tests=0
passed_tests=0

# Test each test case
for i in 0 1 2 3 4 5; do
    f_file="test_cases/f${i}.txt"
    g_file="test_cases/g${i}.txt"
    
    if [ ! -f "$f_file" ] || [ ! -f "$g_file" ]; then
        echo "Test case $i: Files not found, skipping"
        continue
    fi
    
    # Find the expected output file with stride info
    expected=$(ls test_cases/o${i}_*.txt 2>/dev/null | head -1)
    if [ -z "$expected" ]; then
        echo "Test case $i: No expected output found, skipping"
        continue
    fi
    
    # Extract stride values from filename (e.g., o0_sH_1_sW_1.txt -> sH=1, sW=1)
    sH=$(echo "$expected" | sed -n 's/.*_sH_\([0-9]*\)_.*/\1/p')
    sW=$(echo "$expected" | sed -n 's/.*_sW_\([0-9]*\)\.txt/\1/p')
    
    if [ -z "$sH" ] || [ -z "$sW" ]; then
        echo "Test case $i: Could not extract stride values, skipping"
        continue
    fi
    
    echo "Test Case $i (sH=$sH, sW=$sW):"
    
    # Test Sequential
    echo -n "  Sequential: "
    srun -n 1 -c 1 ./a2_performance seq "$f_file" "$g_file" $sH $sW /tmp/out_seq_$i.txt 2>&1 | grep "Time:"
    compare_output "$expected" /tmp/out_seq_$i.txt
    result=$?
    total_tests=$((total_tests + 1))
    if [ $result -eq 0 ]; then passed_tests=$((passed_tests + 1)); fi
    
    # Test OpenMP
    echo -n "  OpenMP (4t): "
    export OMP_NUM_THREADS=4
    srun -n 1 -c 4 ./a2_performance omp "$f_file" "$g_file" $sH $sW /tmp/out_omp_$i.txt 2>&1 | grep "Time:"
    compare_output "$expected" /tmp/out_omp_$i.txt
    result=$?
    total_tests=$((total_tests + 1))
    if [ $result -eq 0 ]; then passed_tests=$((passed_tests + 1)); fi
    
    # Test MPI
    echo -n "  MPI (4p):    "
    srun -n 4 -c 1 ./a2_performance mpi "$f_file" "$g_file" $sH $sW /tmp/out_mpi_$i.txt 2>&1 | grep "Time:"
    compare_output "$expected" /tmp/out_mpi_$i.txt
    result=$?
    total_tests=$((total_tests + 1))
    if [ $result -eq 0 ]; then passed_tests=$((passed_tests + 1)); fi
    
    # Test Hybrid
    echo -n "  Hybrid (2p2t):"
    export OMP_NUM_THREADS=2
    srun -n 2 -c 2 ./a2_performance hybrid "$f_file" "$g_file" $sH $sW /tmp/out_hybrid_$i.txt 2>&1 | grep "Time:"
    compare_output "$expected" /tmp/out_hybrid_$i.txt
    result=$?
    total_tests=$((total_tests + 1))
    if [ $result -eq 0 ]; then passed_tests=$((passed_tests + 1)); fi
    
    echo ""
done

echo "======================================"
echo "Results: $passed_tests / $total_tests tests passed"
echo "======================================"

# Cleanup
rm -f /tmp/out_*.txt
