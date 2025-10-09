#!/usr/bin/env python3
"""
Scaling Analysis and Visualization Script
Analyzes performance scaling results across different matrix sizes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_scaling_data():
    """Load the master scaling results CSV"""
    csv_path = "a2/results2.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def plot_scaling_analysis(df):
    """Create comprehensive scaling analysis plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Performance Scaling Analysis Across Problem Sizes', fontsize=16, fontweight='bold')
    
    matrix_sizes = sorted(df['matrix_size'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(matrix_sizes)))
    
    # 1. Speedup vs Number of Cores (OpenMP)
    ax1 = axes[0, 0]
    for i, size in enumerate(matrix_sizes):
        size_data = df[(df['matrix_size'] == size) & (df['mode'] == 'omp')]
        if not size_data.empty:
            avg_data = size_data.groupby('threads_or_processes')['speedup'].mean()
            ax1.plot(avg_data.index, avg_data.values, 'o-', color=colors[i], 
                    label=f'{size}×{size}', linewidth=2, markersize=6)
    
    # Add ideal speedup line
    max_threads = df[df['mode'] == 'omp']['threads_or_processes'].max()
    ideal_threads = range(1, int(max_threads) + 1)
    ax1.plot(ideal_threads, ideal_threads, 'k--', alpha=0.5, label='Ideal Speedup')
    
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('OpenMP Speedup vs Threads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    
    # 2. Efficiency vs Number of Cores (OpenMP)
    ax2 = axes[0, 1]
    for i, size in enumerate(matrix_sizes):
        size_data = df[(df['matrix_size'] == size) & (df['mode'] == 'omp')]
        if not size_data.empty:
            avg_data = size_data.groupby('threads_or_processes')['efficiency_percent'].mean()
            ax2.plot(avg_data.index, avg_data.values, 'o-', color=colors[i], 
                    label=f'{size}×{size}', linewidth=2, markersize=6)
    
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('OpenMP Efficiency vs Threads')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Speedup vs Number of Processes (MPI)
    ax3 = axes[1, 0]
    for i, size in enumerate(matrix_sizes):
        size_data = df[(df['matrix_size'] == size) & (df['mode'] == 'mpi')]
        if not size_data.empty:
            avg_data = size_data.groupby('threads_or_processes')['speedup'].mean()
            ax3.plot(avg_data.index, avg_data.values, 's-', color=colors[i], 
                    label=f'{size}×{size}', linewidth=2, markersize=6)
    
    # Add ideal speedup line
    max_processes = df[df['mode'] == 'mpi']['threads_or_processes'].max()
    ideal_processes = range(1, int(max_processes) + 1)
    ax3.plot(ideal_processes, ideal_processes, 'k--', alpha=0.5, label='Ideal Speedup')
    
    ax3.set_xlabel('Number of MPI Processes')
    ax3.set_ylabel('Speedup')
    ax3.set_title('MPI Speedup vs Processes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log', base=2)
    
    # 4. Efficiency vs Number of Processes (MPI)
    ax4 = axes[1, 1]
    for i, size in enumerate(matrix_sizes):
        size_data = df[(df['matrix_size'] == size) & (df['mode'] == 'mpi')]
        if not size_data.empty:
            avg_data = size_data.groupby('threads_or_processes')['efficiency_percent'].mean()
            ax4.plot(avg_data.index, avg_data.values, 's-', color=colors[i], 
                    label=f'{size}×{size}', linewidth=2, markersize=6)
    
    ax4.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax4.set_xlabel('Number of MPI Processes')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('MPI Efficiency vs Processes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    
    # 5. Best Performance Comparison Across Sizes
    ax5 = axes[2, 0]
    
    # Find best performance for each mode and size
    best_results = []
    for size in matrix_sizes:
        size_data = df[df['matrix_size'] == size]
        for mode in ['seq', 'omp', 'mpi', 'hybrid']:
            mode_data = size_data[size_data['mode'] == mode]
            if not mode_data.empty:
                best_speedup = mode_data['speedup'].max()
                best_results.append({
                    'matrix_size': size,
                    'mode': mode,
                    'best_speedup': best_speedup
                })
    
    best_df = pd.DataFrame(best_results)
    
    # Plot best speedups
    modes = ['seq', 'omp', 'mpi', 'hybrid']
    mode_colors = ['blue', 'green', 'red', 'purple']
    
    for mode, color in zip(modes, mode_colors):
        mode_data = best_df[best_df['mode'] == mode]
        if not mode_data.empty:
            ax5.plot(mode_data['matrix_size'], mode_data['best_speedup'], 
                    'o-', color=color, label=mode.upper(), linewidth=2, markersize=8)
    
    ax5.set_xlabel('Matrix Size')
    ax5.set_ylabel('Best Speedup Achieved')
    ax5.set_title('Best Speedup by Implementation and Problem Size')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    
    # 6. Runtime vs Problem Size (for best configurations)
    ax6 = axes[2, 1]
    
    # Get best runtime for each size and mode
    runtime_results = []
    for size in matrix_sizes:
        size_data = df[df['matrix_size'] == size]
        for mode in ['seq', 'omp', 'mpi', 'hybrid']:
            mode_data = size_data[size_data['mode'] == mode]
            if not mode_data.empty:
                best_time = mode_data['time_seconds'].min()
                runtime_results.append({
                    'matrix_size': size,
                    'mode': mode,
                    'best_time': best_time
                })
    
    runtime_df = pd.DataFrame(runtime_results)
    
    # Plot runtimes
    for mode, color in zip(modes, mode_colors):
        mode_data = runtime_df[runtime_df['mode'] == mode]
        if not mode_data.empty:
            ax6.plot(mode_data['matrix_size'], mode_data['best_time'], 
                    'o-', color=color, label=mode.upper(), linewidth=2, markersize=8)
    
    ax6.set_xlabel('Matrix Size')
    ax6.set_ylabel('Best Runtime (seconds)')
    ax6.set_title('Best Runtime by Implementation and Problem Size')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('scaling_analysis2.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_efficiency_explanation():
    """Print detailed explanation of efficiency calculation"""
    
    print("="*80)
    print("EFFICIENCY CALCULATION EXPLANATION")
    print("="*80)
    print()
    
    print("1. BASIC DEFINITIONS:")
    print("   • Speedup = Sequential_Time / Parallel_Time")
    print("   • Efficiency = (Speedup / Number_of_Cores) × 100%")
    print()
    
    print("2. WHAT EFFICIENCY MEANS:")
    print("   • Efficiency measures how well we utilize available cores")
    print("   • 100% efficiency = perfect linear scaling (ideal case)")
    print("   • 50% efficiency = each core provides half the expected benefit")
    print("   • >100% efficiency = superlinear speedup (rare, usually due to cache effects)")
    print()
    
    print("3. EXAMPLES:")
    print("   Sequential time: 8.0 seconds")
    print("   4-thread time: 2.0 seconds")
    print("   → Speedup = 8.0 / 2.0 = 4.0x")
    print("   → Efficiency = (4.0 / 4) × 100% = 100% (perfect!)")
    print()
    print("   Sequential time: 8.0 seconds") 
    print("   4-thread time: 3.0 seconds")
    print("   → Speedup = 8.0 / 3.0 = 2.67x")
    print("   → Efficiency = (2.67 / 4) × 100% = 66.7% (typical)")
    print()
    
    print("4. WHY EFFICIENCY DROPS:")
    print("   • Parallelization overhead (thread creation, synchronization)")
    print("   • Communication costs (MPI message passing)")
    print("   • Load imbalance (some cores finish before others)")
    print("   • Memory bandwidth limitations")
    print("   • Cache conflicts and false sharing")
    print("   • Sequential portions of code (Amdahl's Law)")
    print()
    
    print("5. GOOD EFFICIENCY TARGETS:")
    print("   • >80%: Excellent scaling")
    print("   • 60-80%: Good scaling")
    print("   • 40-60%: Acceptable scaling")
    print("   • <40%: Poor scaling (investigate bottlenecks)")
    print()
    
    print("6. MODE-SPECIFIC CONSIDERATIONS:")
    print("   • OpenMP: Shared memory, good for compute-intensive tasks")
    print("   • MPI: Distributed memory, communication overhead increases with processes")
    print("   • Hybrid: Combines benefits and challenges of both approaches")
    print()
    
    print("7. PROBLEM SIZE EFFECTS:")
    print("   • Small problems: Overhead dominates, low efficiency")
    print("   • Large problems: More computation per core, better efficiency")
    print("   • Very large problems: Memory/communication bottlenecks may appear")
    print()
    print("="*80)

def generate_summary_report(df):
    """Generate a summary report of scaling results"""
    
    print("\n" + "="*80)
    print("SCALING PERFORMANCE SUMMARY REPORT")
    print("="*80)
    
    matrix_sizes = sorted(df['matrix_size'].unique())
    
    for size in matrix_sizes:
        print(f"\n{size}×{size} Matrix Results:")
        print("-" * 40)
        
        size_data = df[df['matrix_size'] == size]
        
        # Sequential baseline
        seq_data = size_data[size_data['mode'] == 'seq']
        if not seq_data.empty:
            avg_seq_time = seq_data['time_seconds'].mean()
            print(f"Sequential baseline: {avg_seq_time:.6f} seconds")
        
        # Best results for each mode
        for mode in ['omp', 'mpi', 'hybrid']:
            mode_data = size_data[size_data['mode'] == mode]
            if not mode_data.empty:
                best_speedup = mode_data['speedup'].max()
                best_efficiency = mode_data['efficiency_percent'].max()
                best_speedup_rows = mode_data[mode_data['speedup'] == best_speedup]
                if not best_speedup_rows.empty:
                    best_cores = best_speedup_rows['threads_or_processes'].iloc[0]
                else:
                    best_cores = None  # or set to a default value or handle as needed
                
                print(f"{mode.upper():7} best: {best_speedup:.2f}x speedup, {best_efficiency:.1f}% efficiency ({best_cores} cores)")

def main():
    """Main execution function"""
    
    print("Loading scaling analysis data...")
    df = load_scaling_data()
    
    if df is None:
        print("No data to analyze. Run test_scaling.slurm first!")
        return
    
    print(f"Loaded {len(df)} data points from {len(df['matrix_size'].unique())} matrix sizes")
    
    # Print efficiency explanation
    print_efficiency_explanation()
    
    # Generate summary report
    generate_summary_report(df)
    
    # Create visualizations
    print("\nGenerating scaling analysis plots...")
    plot_scaling_analysis(df)
    
    print("\nAnalysis complete!")
    print("Results saved to: scaling_analysis2.png")

if __name__ == "__main__":
    main()