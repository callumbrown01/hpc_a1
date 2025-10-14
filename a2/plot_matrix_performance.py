#!/usr/bin/env python3
"""
Matrix Performance Analysis Visualization Script
Analyzes and plots performance data from matrix_size.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path):
    """Load and preprocess the performance data"""
    df = pd.read_csv(file_path)
    
    # Create a total cores column for hybrid mode
    df['total_cores'] = df['threads'] * df['procs']
    
    # For non-hybrid modes, use threads for OpenMP and procs for MPI
    df.loc[df['mode'] == 'omp', 'total_cores'] = df.loc[df['mode'] == 'omp', 'threads']
    df.loc[df['mode'] == 'mpi', 'total_cores'] = df.loc[df['mode'] == 'mpi', 'procs']
    df.loc[df['mode'] == 'seq', 'total_cores'] = 1
    
    return df

def plot_speedup_vs_matrix_size(df):
    """Plot speedup vs matrix size for different parallelization modes"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Speedup Analysis Across Matrix Sizes', fontsize=16, fontweight='bold')
    
    matrix_sizes = sorted(df['matrix_size'].unique())
    modes = ['seq', 'omp', 'mpi', 'hybrid']
    mode_titles = ['Sequential', 'OpenMP', 'MPI', 'Hybrid']
    
    for idx, (mode, title) in enumerate(zip(modes, mode_titles)):
        ax = axes[idx // 2, idx % 2]
        
        if mode == 'seq':
            # Sequential is always speedup = 1
            ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Sequential')
            ax.set_ylim(0.8, 1.2)
        else:
            mode_data = df[df['mode'] == mode]
            
            for size in matrix_sizes:
                size_data = mode_data[mode_data['matrix_size'] == size]
                if not size_data.empty:
                    ax.plot(size_data['total_cores'], size_data['speedup'], 
                           marker='o', linewidth=2, label=f'Size {size}')
        
        ax.set_xlabel('Total Cores')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{title} Performance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if mode != 'seq':
            ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    return fig

def plot_best_performance_comparison(df):
    """Plot best speedup achieved for each matrix size and mode"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best speedup per matrix size and mode
    best_speedups = df.groupby(['matrix_size', 'mode'])['speedup'].max().reset_index()
    
    # Pivot for easier plotting
    pivot_speedup = best_speedups.pivot(index='matrix_size', columns='mode', values='speedup')
    
    # Plot 1: Bar chart of best speedups
    pivot_speedup.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Best Speedup by Matrix Size and Mode', fontweight='bold')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Maximum Speedup')
    ax1.legend(title='Parallelization Mode')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Execution time comparison
    best_times = df.groupby(['matrix_size', 'mode'])['time_seconds'].min().reset_index()
    pivot_time = best_times.pivot(index='matrix_size', columns='mode', values='time_seconds')
    
    pivot_time.plot(kind='bar', ax=ax2, logy=True, width=0.8)
    ax2.set_title('Best Execution Time by Matrix Size and Mode', fontweight='bold')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Best Time (seconds, log scale)')
    ax2.legend(title='Parallelization Mode')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_scalability_analysis(df):
    """Plot scalability analysis for each mode"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scalability Analysis: Speedup vs Total Cores', fontsize=16, fontweight='bold')
    
    modes = ['omp', 'mpi', 'hybrid']
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['matrix_size'].unique())))
    
    for idx, mode in enumerate(modes):
        ax = axes[idx // 2, idx % 2]
        mode_data = df[df['mode'] == mode]
        
        for i, size in enumerate(sorted(df['matrix_size'].unique())):
            size_data = mode_data[mode_data['matrix_size'] == size]
            if not size_data.empty:
                cores = size_data['total_cores']
                speedup = size_data['speedup']
                ax.plot(cores, speedup, 'o-', color=colors[i], 
                       linewidth=2, markersize=6, label=f'Size {size}')
        
        # Add ideal speedup line
        max_cores = mode_data['total_cores'].max() if not mode_data.empty else 32
        ideal_cores = np.array([1, 2, 4, 8, 16, 32])
        ideal_cores = ideal_cores[ideal_cores <= max_cores]
        ax.plot(ideal_cores, ideal_cores, 'k--', alpha=0.5, linewidth=2, label='Ideal Speedup')
        
        ax.set_xlabel('Total Cores')
        ax.set_ylabel('Speedup')
        ax.set_title(f'{mode.upper()} Scalability', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
    
    # Efficiency plot in the fourth subplot
    ax = axes[1, 1]
    
    for mode in modes:
        mode_data = df[df['mode'] == mode]
        # Calculate average efficiency across all matrix sizes
        efficiency_data = []
        core_counts = sorted(mode_data['total_cores'].unique())
        
        for cores in core_counts:
            core_data = mode_data[mode_data['total_cores'] == cores]
            if not core_data.empty:
                avg_efficiency = (core_data['speedup'] / cores).mean()
                efficiency_data.append(avg_efficiency)
            else:
                efficiency_data.append(0)
        
        if efficiency_data:
            ax.plot(core_counts, efficiency_data, 'o-', linewidth=2, 
                   markersize=6, label=f'{mode.upper()}')
    
    ax.set_xlabel('Total Cores')
    ax.set_ylabel('Average Efficiency')
    ax.set_title('Average Parallel Efficiency', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    return fig

def plot_heatmap_analysis(df):
    """Create heatmaps showing performance characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Heatmap Analysis', fontsize=16, fontweight='bold')
    
    # Heatmap 1: Speedup by matrix size and total cores (all modes)
    heatmap_data = df.pivot_table(values='speedup', index='matrix_size', 
                                  columns='total_cores', aggfunc='max')
    
    sns.heatmap(heatmap_data, ax=axes[0,0], cmap='YlOrRd', annot=True, 
                fmt='.2f', cbar_kws={'label': 'Speedup'})
    axes[0,0].set_title('Maximum Speedup: Matrix Size vs Total Cores', fontweight='bold')
    axes[0,0].set_xlabel('Total Cores')
    axes[0,0].set_ylabel('Matrix Size')
    
    # Heatmap 2: Best mode by matrix size and core count
    best_mode_data = df.loc[df.groupby(['matrix_size', 'total_cores'])['speedup'].idxmax()]
    mode_pivot = best_mode_data.pivot_table(values='mode', index='matrix_size', 
                                           columns='total_cores', aggfunc='first', fill_value='')
    
    # Convert mode names to numbers for heatmap
    mode_map = {'seq': 0, 'omp': 1, 'mpi': 2, 'hybrid': 3}
    mode_numeric = mode_pivot.applymap(lambda x: mode_map.get(x, -1))
    
    sns.heatmap(mode_numeric, ax=axes[0,1], cmap='tab10', annot=mode_pivot, 
                fmt='s', cbar_kws={'label': 'Best Mode'})
    axes[0,1].set_title('Best Performing Mode: Matrix Size vs Total Cores', fontweight='bold')
    axes[0,1].set_xlabel('Total Cores')
    axes[0,1].set_ylabel('Matrix Size')
    
    # Heatmap 3: Execution time by matrix size and mode
    time_heatmap = df.groupby(['matrix_size', 'mode'])['time_seconds'].min().unstack()
    sns.heatmap(time_heatmap, ax=axes[1,0], cmap='YlOrRd_r', annot=True, 
                fmt='.4f', cbar_kws={'label': 'Time (seconds)'})
    axes[1,0].set_title('Best Execution Time: Matrix Size vs Mode', fontweight='bold')
    axes[1,0].set_xlabel('Mode')
    axes[1,0].set_ylabel('Matrix Size')
    
    # Heatmap 4: Efficiency by matrix size and mode (for specific core counts)
    # Use 8 cores as reference for comparison
    ref_cores = 8
    ref_data = df[df['total_cores'] == ref_cores]
    if not ref_data.empty:
        efficiency_data = ref_data.copy()
        efficiency_data['efficiency'] = efficiency_data['speedup'] / ref_cores
        eff_heatmap = efficiency_data.pivot_table(values='efficiency', 
                                                  index='matrix_size', columns='mode')
        sns.heatmap(eff_heatmap, ax=axes[1,1], cmap='RdYlBu_r', annot=True, 
                    fmt='.3f', cbar_kws={'label': 'Efficiency'})
        axes[1,1].set_title(f'Parallel Efficiency at {ref_cores} Cores', fontweight='bold')
    else:
        axes[1,1].text(0.5, 0.5, 'No data for 8 cores', ha='center', va='center')
        axes[1,1].set_title('Parallel Efficiency (No Data)', fontweight='bold')
    
    axes[1,1].set_xlabel('Mode')
    axes[1,1].set_ylabel('Matrix Size')
    
    plt.tight_layout()
    return fig

def create_summary_statistics(df):
    """Generate summary statistics"""
    print("=== MATRIX PERFORMANCE ANALYSIS SUMMARY ===\n")
    
    print("Dataset Overview:")
    print(f"- Total configurations tested: {len(df)}")
    print(f"- Matrix sizes: {sorted(df['matrix_size'].unique())}")
    print(f"- Parallelization modes: {sorted(df['mode'].unique())}")
    print(f"- Core counts tested: {sorted(df['total_cores'].unique())}")
    print()
    
    print("Best Performance Results:")
    best_overall = df.loc[df['speedup'].idxmax()]
    print(f"- Highest speedup: {best_overall['speedup']:.2f}x")
    print(f"  * Matrix size: {best_overall['matrix_size']}")
    print(f"  * Mode: {best_overall['mode']}")
    print(f"  * Configuration: {best_overall['threads']} threads, {best_overall['procs']} processes")
    print()
    
    print("Performance by Mode:")
    for mode in ['omp', 'mpi', 'hybrid']:
        mode_data = df[df['mode'] == mode]
        if not mode_data.empty:
            best = mode_data.loc[mode_data['speedup'].idxmax()]
            avg_speedup = mode_data['speedup'].mean()
            print(f"- {mode.upper()}:")
            print(f"  * Best speedup: {best['speedup']:.2f}x (size {best['matrix_size']}, {best['total_cores']} cores)")
            print(f"  * Average speedup: {avg_speedup:.2f}x")
    
    print()
    print("Scalability Analysis:")
    for size in sorted(df['matrix_size'].unique()):
        size_data = df[df['matrix_size'] == size]
        best_config = size_data.loc[size_data['speedup'].idxmax()]
        print(f"- Matrix size {size}: Best speedup {best_config['speedup']:.2f}x with {best_config['mode']} mode")

def main():
    """Main function to generate all plots and analysis"""
    # Load data
    print("Loading performance data...")
    df = load_data('a2/matrix_size.csv')
    
    # Generate summary statistics
    create_summary_statistics(df)
    
    # Create plots
    print("\nGenerating performance visualizations...")
    
    # Plot 1: Speedup vs Matrix Size
    fig1 = plot_speedup_vs_matrix_size(df)
    fig1.savefig('speedup_vs_matrix_size.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speedup_vs_matrix_size.png")
    
    # Plot 2: Best Performance Comparison  
    fig2 = plot_best_performance_comparison(df)
    fig2.savefig('best_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: best_performance_comparison.png")
    
    # Plot 3: Scalability Analysis
    fig3 = plot_scalability_analysis(df)
    fig3.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scalability_analysis.png")
    
    # Plot 4: Heatmap Analysis
    fig4 = plot_heatmap_analysis(df)
    fig4.savefig('heatmap_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heatmap_analysis.png")
    
    plt.show()
    print("\nAnalysis complete! All plots have been saved as high-resolution PNG files.")

if __name__ == "__main__":
    main()