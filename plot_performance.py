#!/usr/bin/env python3
"""
Performance Analysis Visualization for 2D Convolution OpenMP Parallelization
Generates comprehensive plots from SLURM job performance data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(filename):
    """Load CSV data and clean it"""
    # Read CSV, skipping comment lines
    df = pd.read_csv(filename, comment='#')
    
    # Remove rows with missing or invalid data
    df = df.dropna(subset=['Sequential_Time', 'Parallel_Time', 'Speedup'])
    df = df[df['Sequential_Time'] > 0]
    df = df[df['Parallel_Time'] > 0]
    df = df[df['Speedup'] > 0]
    
    # Calculate additional metrics
    df['Efficiency'] = df['Speedup'] / df['Threads']
    df['Problem_Size'] = df['Matrix_Size_H'] * df['Matrix_Size_W']
    df['Work_Per_Thread'] = df['Problem_Size'] / df['Threads']
    
    return df

def plot_thread_scaling(df, csv_filename):
    """Plot thread scaling analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter thread scaling data
    thread_data = df[df['Test_Type'] == 'ThreadScaling'].copy()
    
    # Group by matrix size
    sizes = thread_data['Problem_Size'].unique()
    sizes = sorted(sizes)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    # 1. Speedup vs Threads
    for i, size in enumerate(sizes):
        data = thread_data[thread_data['Problem_Size'] == size]
        ax1.plot(data['Threads'], data['Speedup'], 'o-', 
                label=f'{int(np.sqrt(size))}×{int(np.sqrt(size))}', 
                color=colors[i], linewidth=2, markersize=6)
    
    # Add ideal speedup line
    max_threads = thread_data['Threads'].max()
    ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Thread Scaling: Speedup vs Number of Threads')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    
    # 2. Parallel Efficiency
    for i, size in enumerate(sizes):
        data = thread_data[thread_data['Problem_Size'] == size]
        ax2.plot(data['Threads'], data['Efficiency'] * 100, 'o-', 
                color=colors[i], linewidth=2, markersize=6)
    
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax2.axhline(y=50, color='r', linestyle=':', alpha=0.5, label='50% Efficiency')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Thread Scaling: Parallel Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Execution Time vs Threads
    for i, size in enumerate(sizes):
        data = thread_data[thread_data['Problem_Size'] == size]
        ax3.plot(data['Threads'], data['Sequential_Time'], '--', 
                color=colors[i], alpha=0.7, label=f'Sequential ({int(np.sqrt(size))}×{int(np.sqrt(size))})')
        ax3.plot(data['Threads'], data['Parallel_Time'], 'o-', 
                color=colors[i], linewidth=2, markersize=6, 
                label=f'Parallel ({int(np.sqrt(size))}×{int(np.sqrt(size))})')
    
    ax3.set_xlabel('Number of Threads')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Thread Scaling: Execution Time')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    
    # 4. Speedup Heatmap
    pivot_data = thread_data.pivot(index='Problem_Description', columns='Threads', values='Speedup')
    im = ax4.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(pivot_data.columns)))
    ax4.set_xticklabels(pivot_data.columns)
    ax4.set_yticks(range(len(pivot_data.index)))
    ax4.set_yticklabels([desc.replace('Size_', '').replace('x', '×') for desc in pivot_data.index])
    ax4.set_xlabel('Number of Threads')
    ax4.set_ylabel('Matrix Size')
    ax4.set_title('Speedup Heatmap: Thread Scaling')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Speedup Factor')
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if not pd.isna(pivot_data.iloc[i, j]):
                text = ax4.text(j, i, f'{pivot_data.iloc[i, j]:.1f}', 
                               ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    import os
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'thread_scaling_analysis.png'), dpi=300, bbox_inches='tight')

def plot_problem_size_scaling(df, csv_filename):
    """Plot problem size scaling analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Matrix size scaling data
    matrix_data = df[df['Test_Type'] == 'MatrixScaling'].copy()
    
    # 1. Speedup vs Problem Size
    ax1.plot(matrix_data['Problem_Size'], matrix_data['Speedup'], 'bo-', 
             linewidth=3, markersize=8, label='16 Threads')
    ax1.set_xlabel('Problem Size (Matrix Elements)')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Problem Size Scaling: Speedup vs Matrix Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.legend()
    
    # Add annotations for key points
    for _, row in matrix_data.iterrows():
        ax1.annotate(f'{int(np.sqrt(row["Problem_Size"]))}²', 
                    (row['Problem_Size'], row['Speedup']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 2. Execution Time vs Problem Size
    ax2.plot(matrix_data['Problem_Size'], matrix_data['Sequential_Time'], 
             's--', color='red', linewidth=2, markersize=6, label='Sequential')
    ax2.plot(matrix_data['Problem_Size'], matrix_data['Parallel_Time'], 
             'o-', color='blue', linewidth=3, markersize=8, label='Parallel (16 threads)')
    ax2.set_xlabel('Problem Size (Matrix Elements)')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Problem Size Scaling: Execution Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    
    # 3. Kernel Size Scaling
    kernel_data = df[df['Test_Type'] == 'KernelScaling'].copy()
    
    ax3.plot(kernel_data['Kernel_Size'], kernel_data['Speedup'], 'go-', 
             linewidth=3, markersize=8, label='1000×1000, 16 threads')
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('Speedup')
    ax3.set_title('Kernel Size Scaling: Speedup vs Kernel Size')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add trend line
    z = np.polyfit(kernel_data['Kernel_Size'], kernel_data['Speedup'], 1)
    p = np.poly1d(z)
    ax3.plot(kernel_data['Kernel_Size'], p(kernel_data['Kernel_Size']), "r--", alpha=0.8)
    
    # 4. Efficiency vs Work per Thread
    all_data = df[df['Test_Type'].isin(['ThreadScaling', 'MatrixScaling'])].copy()
    
    scatter = ax4.scatter(all_data['Work_Per_Thread'], all_data['Efficiency'] * 100, 
                         c=all_data['Threads'], s=60, alpha=0.7, cmap='plasma')
    ax4.set_xlabel('Work per Thread (Matrix Elements)')
    ax4.set_ylabel('Parallel Efficiency (%)')
    ax4.set_title('Efficiency vs Work Distribution')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # Add efficiency thresholds
    ax4.axhline(y=50, color='red', linestyle=':', alpha=0.7, label='50% Efficiency')
    ax4.axhline(y=80, color='orange', linestyle=':', alpha=0.7, label='80% Efficiency')
    ax4.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Number of Threads')
    
    plt.tight_layout()
    import os
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'problem_size_scaling.png'), dpi=300, bbox_inches='tight')

def plot_comprehensive_analysis(df, csv_filename):
    """Plot comprehensive performance matrix analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Comprehensive data
    comp_data = df[df['Test_Type'] == 'Comprehensive'].copy()
    
    # Extract matrix size and kernel size from description
    comp_data['Matrix_Size'] = comp_data['Problem_Description'].str.extract(r'M(\d+)').astype(int)
    comp_data['Kernel_Size'] = comp_data['Problem_Description'].str.extract(r'K(\d+)').astype(int)
    
    # 1. 3D Surface-like plot: Speedup vs Matrix Size vs Thread Count (for K=5)
    k5_data = comp_data[comp_data['Kernel_Size'] == 5]
    pivot_k5 = k5_data.pivot(index='Matrix_Size', columns='Threads', values='Speedup')
    
    im1 = ax1.imshow(pivot_k5.values, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_xticks(range(len(pivot_k5.columns)))
    ax1.set_xticklabels(pivot_k5.columns)
    ax1.set_yticks(range(len(pivot_k5.index)))
    ax1.set_yticklabels([f'{size}×{size}' for size in pivot_k5.index])
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Matrix Size')
    ax1.set_title('Speedup Heatmap: Matrix Size vs Threads (5×5 kernel)')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Speedup Factor')
    
    # 2. Peak Performance Analysis
    best_speedups = comp_data.groupby(['Matrix_Size', 'Kernel_Size'])['Speedup'].max().reset_index()
    best_configs = comp_data.merge(best_speedups, on=['Matrix_Size', 'Kernel_Size', 'Speedup'])
    
    scatter2 = ax2.scatter(best_configs['Matrix_Size'], best_configs['Speedup'], 
                          c=best_configs['Kernel_Size'], s=best_configs['Threads']*3, 
                          alpha=0.7, cmap='coolwarm')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Best Speedup Achieved')
    ax2.set_title('Peak Performance: Best Speedup by Problem Size')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Kernel Size')
    
    # Add legend for point sizes
    for threads in [16, 32, 48, 64]:
        if threads in best_configs['Threads'].values:
            ax2.scatter([], [], s=threads*3, c='gray', alpha=0.7, 
                       label=f'{threads} threads')
    ax2.legend(title='Optimal Thread Count', loc='lower right')
    
    # 3. Performance vs Kernel Size for different matrix sizes
    matrix_sizes = [500, 1000, 1500]
    colors = ['red', 'blue', 'green']
    
    for i, mat_size in enumerate(matrix_sizes):
        size_data = comp_data[(comp_data['Matrix_Size'] == mat_size) & (comp_data['Threads'] == 48)]
        if not size_data.empty:
            ax3.plot(size_data['Kernel_Size'], size_data['Speedup'], 'o-', 
                    color=colors[i], linewidth=2, markersize=6, 
                    label=f'{mat_size}×{mat_size} matrix')
    
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('Speedup (48 threads)')
    ax3.set_title('Kernel Size Impact on Performance')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Thread Efficiency Analysis across all configurations
    efficiency_data = comp_data.copy()
    
    # Create efficiency categories
    efficiency_data['Efficiency_Category'] = pd.cut(efficiency_data['Efficiency'], 
                                                   bins=[0, 0.25, 0.5, 0.75, 1.0, float('inf')],
                                                   labels=['Poor (<25%)', 'Fair (25-50%)', 
                                                          'Good (50-75%)', 'Excellent (75-100%)', 
                                                          'Super-linear (>100%)'])
    
    # Count configurations in each efficiency category
    eff_counts = efficiency_data['Efficiency_Category'].value_counts()
    
    wedges, texts, autotexts = ax4.pie(eff_counts.values, labels=eff_counts.index, 
                                      autopct='%1.1f%%', startangle=90, 
                                      colors=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'])
    ax4.set_title('Distribution of Parallel Efficiency\n(All Configurations)')
    
    # Enhance pie chart appearance
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    import os
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')

def plot_performance_summary(df, csv_filename):
    """Create a summary dashboard"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a complex grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Key metrics
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:])
    ax5 = fig.add_subplot(gs[2, :])
    
    # 1. Best speedups achieved
    best_overall = df.groupby('Test_Type')['Speedup'].max()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars1 = ax1.bar(range(len(best_overall)), best_overall.values, color=colors)
    ax1.set_xticks(range(len(best_overall)))
    ax1.set_xticklabels(best_overall.index, rotation=45)
    ax1.set_ylabel('Maximum Speedup')
    ax1.set_title('Peak Performance by Test Category')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, best_overall.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 2. Thread utilization efficiency
    thread_data = df[df['Test_Type'] == 'ThreadScaling']
    
    # Average efficiency by thread count
    avg_efficiency = thread_data.groupby('Threads')['Efficiency'].mean() * 100
    
    ax2.plot(avg_efficiency.index, avg_efficiency.values, 'ro-', linewidth=3, markersize=8)
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Efficiency')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Average Efficiency (%)')
    ax2.set_title('Thread Utilization Efficiency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xscale('log', base=2)
    
    # 3. Problem size sweet spot
    matrix_data = df[df['Test_Type'] == 'MatrixScaling']
    
    ax3.plot(matrix_data['Problem_Size'], matrix_data['Efficiency'] * 100, 'bs-', 
             linewidth=2, markersize=8)
    ax3.set_xlabel('Problem Size (Matrix Elements)')
    ax3.set_ylabel('Efficiency (%) at 16 threads')
    ax3.set_title('Efficiency vs Problem Size')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Find and mark the sweet spot
    max_eff_idx = matrix_data['Efficiency'].idxmax()
    max_eff_point = matrix_data.loc[max_eff_idx]
    ax3.plot(max_eff_point['Problem_Size'], max_eff_point['Efficiency'] * 100, 
             'ro', markersize=12, markerfacecolor='red', markeredgecolor='darkred')
    ax3.annotate(f'Sweet Spot\n{int(np.sqrt(max_eff_point["Problem_Size"]))}×{int(np.sqrt(max_eff_point["Problem_Size"]))}',
                xy=(max_eff_point['Problem_Size'], max_eff_point['Efficiency'] * 100),
                xytext=(20, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 4. Kernel size impact
    kernel_data = df[df['Test_Type'] == 'KernelScaling']
    
    # Calculate computational intensity (operations per element)
    kernel_data_copy = kernel_data.copy()
    kernel_data_copy['Ops_Per_Element'] = kernel_data_copy['Kernel_Size'] ** 2
    
    ax4.scatter(kernel_data_copy['Ops_Per_Element'], kernel_data_copy['Speedup'], 
               c=kernel_data_copy['Kernel_Size'], s=100, alpha=0.7, cmap='plasma')
    ax4.set_xlabel('Operations per Output Element')
    ax4.set_ylabel('Speedup (16 threads)')
    ax4.set_title('Computational Intensity vs Speedup')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(kernel_data_copy['Ops_Per_Element'], kernel_data_copy['Speedup'], 1)
    p = np.poly1d(z)
    ax4.plot(kernel_data_copy['Ops_Per_Element'], p(kernel_data_copy['Ops_Per_Element']), 
             "r--", alpha=0.8, linewidth=2)
    
    # 5. Performance timeline/overview
    all_data = df[df['Test_Type'].isin(['ThreadScaling', 'Comprehensive'])]
    
    # Create a comprehensive performance map
    perf_summary = []
    
    for test_type in ['ThreadScaling', 'MatrixScaling', 'KernelScaling', 'Comprehensive']:
        test_data = df[df['Test_Type'] == test_type]
        if not test_data.empty:
            perf_summary.append({
                'Test': test_type,
                'Max_Speedup': test_data['Speedup'].max(),
                'Avg_Speedup': test_data['Speedup'].mean(),
                'Best_Efficiency': test_data['Efficiency'].max() * 100,
                'Configurations': len(test_data)
            })
    
    summary_df = pd.DataFrame(perf_summary)
    
    # Create a multi-metric comparison
    x = np.arange(len(summary_df))
    width = 0.2
    
    bars1 = ax5.bar(x - width, summary_df['Max_Speedup'], width, 
                    label='Max Speedup', color='lightblue', alpha=0.8)
    bars2 = ax5.bar(x, summary_df['Avg_Speedup'], width, 
                    label='Avg Speedup', color='lightgreen', alpha=0.8)
    bars3 = ax5.bar(x + width, summary_df['Best_Efficiency']/10, width, 
                    label='Best Efficiency/10', color='lightcoral', alpha=0.8)
    
    ax5.set_xlabel('Test Category')
    ax5.set_ylabel('Performance Metrics')
    ax5.set_title('Performance Summary Across All Test Categories')
    ax5.set_xticks(x)
    ax5.set_xticklabels(summary_df['Test'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('OpenMP 2D Convolution Performance Analysis Dashboard', 
                fontsize=16, fontweight='bold')
    import os
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')

def generate_performance_report(df):
    """Generate a text-based performance report"""
    print("=" * 80)
    print("OpenMP 2D CONVOLUTION PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"Total configurations tested: {len(df)}")
    print(f"Maximum speedup achieved: {df['Speedup'].max():.2f}x")
    print(f"Average speedup across all tests: {df['Speedup'].mean():.2f}x")
    print(f"Best parallel efficiency: {df['Efficiency'].max()*100:.1f}%")
    
    # Thread scaling insights
    thread_data = df[df['Test_Type'] == 'ThreadScaling']
    print(f"\n🧵 THREAD SCALING INSIGHTS")
    print(f"Best thread scaling speedup: {thread_data['Speedup'].max():.2f}x")
    
    # Find optimal thread count
    best_thread_config = thread_data.loc[thread_data['Speedup'].idxmax()]
    print(f"Optimal configuration: {int(np.sqrt(best_thread_config['Problem_Size']))}×{int(np.sqrt(best_thread_config['Problem_Size']))} matrix with {best_thread_config['Threads']} threads")
    
    # Efficiency analysis
    high_eff = thread_data[thread_data['Efficiency'] > 0.5]
    print(f"Configurations with >50% efficiency: {len(high_eff)}/{len(thread_data)} ({len(high_eff)/len(thread_data)*100:.1f}%)")
    
    # Problem size insights
    matrix_data = df[df['Test_Type'] == 'MatrixScaling']
    print(f"\n📏 PROBLEM SIZE INSIGHTS")
    if 'Problem_Size' in matrix_data.columns and not matrix_data.empty:
        sweet_spot = matrix_data.loc[matrix_data['Efficiency'].idxmax()]
        print(f"Efficiency sweet spot: {int(np.sqrt(sweet_spot['Problem_Size']))}×{int(np.sqrt(sweet_spot['Problem_Size']))} matrix ({sweet_spot['Efficiency']*100:.1f}% efficiency)")
    else:
        print("No 'Problem_Size' data available for MatrixScaling.")
    
    # Kernel size insights
    kernel_data = df[df['Test_Type'] == 'KernelScaling']
    print(f"\n🔍 KERNEL SIZE INSIGHTS")
    if not kernel_data.empty and kernel_data['Speedup'].notna().any():
        best_idx = kernel_data['Speedup'].idxmax()
        if best_idx in kernel_data.index:
            best_kernel_size = kernel_data.loc[best_idx, 'Kernel_Size']
            best_speedup = kernel_data.loc[best_idx, 'Speedup']
            print(f"Best kernel size performance: {best_kernel_size}×{best_kernel_size} kernel ({best_speedup:.2f}x speedup)")
        else:
            print("No valid kernel scaling data available.")
    else:
        print("No kernel scaling data available.")
    
    # Scalability analysis
    print(f"\n📈 SCALABILITY ANALYSIS")
    if 'Problem_Size' in thread_data.columns:
        # Find configurations that scale well (efficiency > 25% at high thread counts)
        high_thread_data = thread_data[thread_data['Threads'] >= 32]
        good_scaling = high_thread_data[high_thread_data['Efficiency'] > 0.25]
        print(f"Configurations scaling well to 32+ threads: {len(good_scaling)}")
        if len(good_scaling) > 0:
            print("Well-scaling problem sizes:")
            for _, row in good_scaling.groupby('Problem_Size').first().iterrows():
                if 'Problem_Size' in row:
                    size = int(np.sqrt(row['Problem_Size']))
                    print(f"  - {size}×{size} matrices")
                else:
                    print("  - Problem size data not available for this configuration.")
    else:
        print("No 'Problem_Size' data available for scalability analysis.")
    
    print("\n" + "=" * 80)

def main():
    """Main function to generate all plots and analysis"""
    print("Loading performance data...")
    
    # Try multiple possible filenames
    filenames = ['collapse(2).csv']
    df = None
    
    csv_filename = None
    for filename in filenames:
        try:
            df = load_and_clean_data(filename)
            csv_filename = filename
            print(f"Successfully loaded data from {filename}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if df is None or csv_filename is None:
        print("Error: Could not find performance data file!")
        print("Expected files: " + ", ".join(filenames))
        return

    print(f"Loaded {len(df)} data points")
    print("\nGenerating visualizations...")

    # Generate all plots
    try:
        print("1. Thread scaling analysis...")
        plot_thread_scaling(df, csv_filename)
        
        print("2. Problem size scaling analysis...")
        plot_problem_size_scaling(df, csv_filename)
        
        print("3. Comprehensive analysis...")
        plot_comprehensive_analysis(df, csv_filename)
        
        print("4. Performance dashboard...")
        plot_performance_summary(df, csv_filename)
        
        print("5. Generating performance report...")
        generate_performance_report(df)
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

    base = csv_filename.rsplit('.', 1)[0]
    print("\n Analysis complete! Generated files:")
    print(f"  - {base}/thread_scaling_analysis.png")
    print(f"  - {base}/problem_size_scaling.png")
    print(f"  - {base}/comprehensive_analysis.png")
    print(f"  - {base}/performance_dashboard.png")

if __name__ == "__main__":
    main()
