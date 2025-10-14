#!/usr/bin/env python3
"""
Focused Performance Analysis for 2D Convolution OpenMP Parallelization
Generates specific performance plots as requested
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(filename):
    """Load CSV data and clean it"""
    try:
        # Read CSV, skipping comment lines
        df = pd.read_csv(filename, comment='#')
        print(f"Successfully read CSV with {len(df)} rows")
        
        # Data is already in the correct format, just validate required columns exist
        required_cols = ['Test_Category', 'Matrix_H', 'Matrix_W', 'Kernel_H', 'Kernel_W', 
                         'Threads', 'Sequential_Time', 'Best_Time', 'Speedup', 'Best_Algorithm']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"All required columns present: {required_cols}")
        
        # Remove rows with missing or invalid data
        df = df.dropna(subset=['Sequential_Time', 'Best_Time', 'Speedup'])
        df = df[df['Sequential_Time'] > 0]
        df = df[df['Best_Time'] > 0]
        df = df[df['Speedup'] > 0]
        
        print(f"After filtering: {len(df)} rows")
        
        # Calculate additional metrics if not present
        if 'Efficiency' not in df.columns:
            df['Efficiency'] = df['Speedup'] / df['Threads']
        else:
            # Convert efficiency to numeric if it's not already
            df['Efficiency'] = pd.to_numeric(df['Efficiency'], errors='coerce')
        
        # Always recalculate Problem_Size from Matrix dimensions to ensure it's numeric
        df['Problem_Size'] = df['Matrix_H'] * df['Matrix_W']
        
        # Convert threads to numeric
        df['Threads'] = pd.to_numeric(df['Threads'], errors='coerce')
        
        # Now calculate Work_Per_Thread with numeric values
        df['Work_Per_Thread'] = df['Problem_Size'] / df['Threads']
        df['Parallel_Time'] = df['Best_Time']  # Use best time as parallel time
        
        print("Data loaded and processed successfully")
        return df
        
    except Exception as e:
        print(f"Error in load_and_clean_data: {e}")
        import traceback
        traceback.print_exc()
        raise

def plot_algorithm_performance_comparison(df, csv_filename):
    """Plot algorithm performance comparison and speedup analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Algorithm comparison data
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison'].copy()
    
    # 1. Algorithm Performance Comparison (Average Speedup)
    if not algo_data.empty:
        algo_perf = algo_data.groupby('Best_Algorithm')['Speedup'].mean().sort_values(ascending=False)
        
        bars = ax1.bar(range(len(algo_perf)), algo_perf.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'][:len(algo_perf)])
        ax1.set_xticks(range(len(algo_perf)))
        ax1.set_xticklabels(algo_perf.index, rotation=45)
        ax1.set_ylabel('Average Speedup')
        ax1.set_title('Algorithm Performance Comparison (Average Speedup)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, algo_perf.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 2. Speedup vs Number of Threads for Each Algorithm Type
    if not algo_data.empty:
        # Calculate speedup for each algorithm type using individual timing columns
        algorithms = []
        algo_names = []
        colors = ['red', 'blue', 'green', 'orange']
        
        # Check which algorithm columns exist
        if 'Basic_Static_Time' in algo_data.columns:
            algorithms.append('Basic_Static_Time')
            algo_names.append('Basic Static')
        if 'Dynamic_Time' in algo_data.columns:
            algorithms.append('Dynamic_Time')
            algo_names.append('Dynamic')
        if 'Guided_Time' in algo_data.columns:
            algorithms.append('Guided_Time')
            algo_names.append('Guided')
        if 'Collapse_Time' in algo_data.columns:
            algorithms.append('Collapse_Time')
            algo_names.append('Collapse')
        
        for i, (algo_col, algo_name, color) in enumerate(zip(algorithms, algo_names, colors)):
            if algo_col in algo_data.columns:
                # Calculate speedup for this algorithm
                algo_data[f'{algo_name}_Speedup'] = algo_data['Sequential_Time'] / algo_data[algo_col]
                
                # Group by threads and get average speedup
                thread_speedup = algo_data.groupby('Threads')[f'{algo_name}_Speedup'].mean()
                
                ax2.plot(thread_speedup.index, thread_speedup.values, 'o-', 
                        linewidth=2, markersize=6, label=algo_name, color=color)
        
        # Add ideal speedup line
        max_threads = algo_data['Threads'].max()
        ax2.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
        ax2.set_xlabel('Number of Threads')
        ax2.set_ylabel('Average Speedup')
        ax2.set_title('Speedup vs Number of Threads by Algorithm Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log', base=2)
    
    # 3. Thread Utilization Efficiency
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    if not thread_data.empty:
        # Average efficiency by thread count
        avg_efficiency = thread_data.groupby('Threads')['Efficiency'].mean() * 100
        
        ax3.plot(avg_efficiency.index, avg_efficiency.values, 'ro-', linewidth=3, markersize=8)
        ax3.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Efficiency')
        ax3.set_xlabel('Number of Threads')
        ax3.set_ylabel('Average Efficiency (%)')
        ax3.set_title('Thread Utilization Efficiency')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xscale('log', base=2)
    
    # 4. Speedup Heatmap for Thread Scaling
    if not thread_data.empty:
        try:
            # Create speedup heatmap
            pivot_speedup = thread_data.pivot_table(
                index='Problem_Size', 
                columns='Threads', 
                values='Speedup', 
                aggfunc='mean'
            )
            
            if not pivot_speedup.empty:
                im = ax4.imshow(pivot_speedup.values, cmap='YlOrRd', aspect='auto')
                ax4.set_xticks(range(len(pivot_speedup.columns)))
                ax4.set_xticklabels(pivot_speedup.columns)
                ax4.set_yticks(range(len(pivot_speedup.index)))
                ax4.set_yticklabels([f'{int(np.sqrt(size))}²' for size in pivot_speedup.index])
                ax4.set_xlabel('Number of Threads')
                ax4.set_ylabel('Matrix Size')
                ax4.set_title('Speedup Heatmap: Thread Scaling')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Speedup Factor')
                
                # Add text annotations
                for i in range(len(pivot_speedup.index)):
                    for j in range(len(pivot_speedup.columns)):
                        if not pd.isna(pivot_speedup.iloc[i, j]):
                            speedup_val = pivot_speedup.iloc[i, j]
                            color = 'white' if speedup_val < 10 else 'black'
                            ax4.text(j, i, f'{speedup_val:.1f}', 
                                   ha="center", va="center", color=color, fontweight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Heatmap data unavailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Speedup Heatmap: Thread Scaling')
    
    plt.tight_layout()
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'algorithm_performance_analysis.png'), dpi=300, bbox_inches='tight')

def plot_best_algorithm_heatmap(df, csv_filename):
    """Plot heatmap showing best algorithm for each parameter combination"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison'].copy()
    
    if not algo_data.empty:
        # 1. Best Algorithm Heatmap by Matrix Size and Kernel Size
        try:
            # Create a pivot table for best algorithm
            pivot_algo = algo_data.pivot_table(
                index='Kernel_H', 
                columns='Matrix_H', 
                values='Best_Algorithm', 
                aggfunc=lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
            )
            
            # Create numerical mapping for algorithms
            unique_algos = algo_data['Best_Algorithm'].unique()
            algo_map = {algo: i for i, algo in enumerate(unique_algos)}
            
            # Convert to numerical values
            pivot_numeric = pivot_algo.map(algo_map)
            
            im1 = ax1.imshow(pivot_numeric.values, cmap='tab10', aspect='auto')
            ax1.set_xticks(range(len(pivot_algo.columns)))
            ax1.set_xticklabels([f'{size}×{size}' for size in pivot_algo.columns])
            ax1.set_yticks(range(len(pivot_algo.index)))
            ax1.set_yticklabels([f'{size}×{size}' for size in pivot_algo.index])
            ax1.set_xlabel('Matrix Size')
            ax1.set_ylabel('Kernel Size')
            ax1.set_title('Best Algorithm by Matrix and Kernel Size')
            
            # Add text annotations
            for i in range(len(pivot_algo.index)):
                for j in range(len(pivot_algo.columns)):
                    if not pd.isna(pivot_numeric.iloc[i, j]):
                        algo_name = pivot_algo.iloc[i, j]
                        ax1.text(j, i, algo_name.replace('scheduling', '').replace('approach', ''), 
                               ha="center", va="center", color='white', fontweight='bold', fontsize=8)
            
        except Exception as e:
            ax1.text(0.5, 0.5, 'Best Algorithm Heatmap unavailable', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Best Algorithm by Matrix and Kernel Size')
        
        # 2. Best Algorithm Heatmap by Thread Count and Problem Size
        try:
            pivot_algo_threads = algo_data.pivot_table(
                index='Problem_Size', 
                columns='Threads', 
                values='Best_Algorithm', 
                aggfunc=lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
            )
            
            # Convert to numerical values
            pivot_numeric_threads = pivot_algo_threads.map(algo_map)
            
            im2 = ax2.imshow(pivot_numeric_threads.values, cmap='tab10', aspect='auto')
            ax2.set_xticks(range(len(pivot_algo_threads.columns)))
            ax2.set_xticklabels(pivot_algo_threads.columns)
            ax2.set_yticks(range(len(pivot_algo_threads.index)))
            ax2.set_yticklabels([f'{int(np.sqrt(size))}²' for size in pivot_algo_threads.index])
            ax2.set_xlabel('Number of Threads')
            ax2.set_ylabel('Matrix Size')
            ax2.set_title('Best Algorithm by Matrix Size and Thread Count')
            
            # Add text annotations
            for i in range(len(pivot_algo_threads.index)):
                for j in range(len(pivot_algo_threads.columns)):
                    if not pd.isna(pivot_numeric_threads.iloc[i, j]):
                        algo_name = pivot_algo_threads.iloc[i, j]
                        ax2.text(j, i, algo_name.replace('scheduling', '').replace('approach', ''), 
                               ha="center", va="center", color='white', fontweight='bold', fontsize=8)
                        
        except Exception as e:
            ax2.text(0.5, 0.5, 'Best Algorithm Heatmap unavailable', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Best Algorithm by Matrix Size and Thread Count')
    
    plt.tight_layout()
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'best_algorithm_heatmaps.png'), dpi=300, bbox_inches='tight')

def plot_kernel_and_size_impact(df, csv_filename):
    """Plot kernel size impact and matrix size scaling analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Kernel Size Impact on Performance
    kernel_data = df[df['Test_Category'] == 'Kernel_Impact'].copy()
    
    if not kernel_data.empty:
        # Group by kernel size and find best speedup for each
        kernel_sizes = sorted(kernel_data['Kernel_H'].unique())
        speedups = []
        
        for k_size in kernel_sizes:
            k_data = kernel_data[kernel_data['Kernel_H'] == k_size]
            if not k_data.empty:
                speedups.append(k_data['Speedup'].max())
            else:
                speedups.append(0)
        
        ax1.plot(kernel_sizes, speedups, 'go-', linewidth=3, markersize=8, label='Best Speedup')
        ax1.set_xlabel('Kernel Size')
        ax1.set_ylabel('Maximum Speedup Achieved')
        ax1.set_title('Kernel Size Impact on Performance')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add trend line
        if len(kernel_sizes) > 1:
            z = np.polyfit(kernel_sizes, speedups, 1)
            p = np.poly1d(z)
            ax1.plot(kernel_sizes, p(kernel_sizes), "r--", alpha=0.8, linewidth=2)
    
    # 2. Speedup vs Matrix Size
    all_data = df[df['Test_Category'].isin(['Algorithm_Comparison', 'Thread_Scalability', 'Kernel_Impact'])].copy()
    
    if not all_data.empty:
        # Group by matrix size and get max speedup
        matrix_speedups = all_data.groupby('Matrix_H')['Speedup'].max()
        matrix_sizes = matrix_speedups.index
        
        ax2.plot(matrix_sizes, matrix_speedups.values, 'bo-', linewidth=3, markersize=8)
        ax2.set_xlabel('Matrix Size (N×N)')
        ax2.set_ylabel('Maximum Speedup Achieved')
        ax2.set_title('Speedup vs Matrix Size')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Annotate peak performance points
        for size, speedup in zip(matrix_sizes, matrix_speedups.values):
            ax2.annotate(f'{speedup:.1f}x', (size, speedup), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Execution Time vs Problem Size
    if not all_data.empty:
        # Plot execution times for different thread counts
        thread_counts = [1, 8, 16, 32, 48]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, threads in enumerate(thread_counts):
            thread_subset = all_data[all_data['Threads'] == threads]
            if not thread_subset.empty:
                # Group by problem size
                size_times = thread_subset.groupby('Problem_Size').agg({
                    'Sequential_Time': 'mean',
                    'Parallel_Time': 'mean'
                })
                
                if threads == 1:
                    ax3.plot(size_times.index, size_times['Sequential_Time'], 's--', 
                            color='black', linewidth=2, markersize=6, label='Sequential')
                else:
                    ax3.plot(size_times.index, size_times['Parallel_Time'], 'o-', 
                            color=colors[i], linewidth=2, markersize=6, label=f'{threads} threads')
        
        ax3.set_xlabel('Problem Size (Matrix Elements)')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Execution Time vs Problem Size vs Thread Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
    
    # 4. Thread Scaling by Problem Size
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    if not thread_data.empty:
        # Group by problem size
        sizes = sorted(thread_data['Problem_Size'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
        
        for i, size in enumerate(sizes):
            data = thread_data[thread_data['Problem_Size'] == size]
            if not data.empty:
                matrix_size = int(np.sqrt(size))
                ax4.plot(data['Threads'], data['Speedup'], 'o-', 
                        label=f'{matrix_size}×{matrix_size}', 
                        color=colors[i], linewidth=2, markersize=6)
        
        # Add ideal speedup line
        max_threads = thread_data['Threads'].max()
        ax4.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal')
        ax4.set_xlabel('Number of Threads')
        ax4.set_ylabel('Speedup')
        ax4.set_title('Thread Scaling by Problem Size')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        ax4.set_yscale('log', base=2)
    
    plt.tight_layout()
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'kernel_and_size_impact.png'), dpi=300, bbox_inches='tight')

def generate_focused_report(df):
    """Generate a focused performance report"""
    print("=" * 80)
    print("FOCUSED OpenMP 2D CONVOLUTION PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print(f"Total configurations tested: {len(df)}")
    print(f"Maximum speedup achieved: {df['Speedup'].max():.2f}x")
    print(f"Average speedup across all tests: {df['Speedup'].mean():.2f}x")
    print(f"Best parallel efficiency: {df['Efficiency'].max()*100:.1f}%")
    
    # Algorithm performance insights
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        print(f"\n🔬 ALGORITHM PERFORMANCE")
        best_algo = algo_data.groupby('Best_Algorithm')['Speedup'].mean().idxmax()
        best_algo_speedup = algo_data.groupby('Best_Algorithm')['Speedup'].mean().max()
        print(f"Best performing algorithm: {best_algo} (avg {best_algo_speedup:.2f}x speedup)")
        
        print("Algorithm performance ranking:")
        algo_ranking = algo_data.groupby('Best_Algorithm')['Speedup'].mean().sort_values(ascending=False)
        for i, (algo, speedup) in enumerate(algo_ranking.items(), 1):
            print(f"  {i}. {algo}: {speedup:.2f}x average speedup")
    
    # Thread scaling insights
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    if not thread_data.empty:
        print(f"\n🧵 THREAD SCALING INSIGHTS")
        print(f"Best thread scaling speedup: {thread_data['Speedup'].max():.2f}x")
        
        # Find optimal configuration
        best_config = thread_data.loc[thread_data['Speedup'].idxmax()]
        if 'Problem_Size' in best_config:
            matrix_size = int(np.sqrt(best_config['Problem_Size']))
            print(f"Optimal configuration: {matrix_size}×{matrix_size} matrix with {best_config['Threads']} threads")
        
        # Efficiency at different thread counts
        eff_32 = thread_data[thread_data['Threads'] == 32]['Efficiency'].mean() * 100 if 32 in thread_data['Threads'].values else 0
        eff_48 = thread_data[thread_data['Threads'] == 48]['Efficiency'].mean() * 100 if 48 in thread_data['Threads'].values else 0
        print(f"Average efficiency at 32 threads: {eff_32:.1f}%")
        print(f"Average efficiency at 48 threads: {eff_48:.1f}%")
    
    # Kernel size insights
    kernel_data = df[df['Test_Category'] == 'Kernel_Impact']
    if not kernel_data.empty:
        print(f"\n🔍 KERNEL SIZE INSIGHTS")
        best_kernel = kernel_data.loc[kernel_data['Speedup'].idxmax()]
        print(f"Best kernel performance: {best_kernel['Kernel_H']}×{best_kernel['Kernel_W']} kernel ({best_kernel['Speedup']:.2f}x speedup)")
        
        kernel_performance = kernel_data.groupby('Kernel_H')['Speedup'].max().sort_values(ascending=False)
        print("Kernel size performance ranking:")
        for size, speedup in kernel_performance.items():
            print(f"  {size}×{size}: {speedup:.2f}x max speedup")
    
    print("\n" + "=" * 80)

def main():
    """Main function to generate focused plots and analysis"""
    print("Loading performance data...")
    
    # Try the CSV file
    csv_filename = 'performance_test.csv'
    try:
        df = load_and_clean_data(csv_filename)
        print(f"Successfully loaded data from {csv_filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}!")
        return
    except Exception as e:
        print(f"Error loading {csv_filename}: {e}")
        return
    
    print(f"Loaded {len(df)} data points")
    print(f"Available test categories: {df['Test_Category'].unique()}")
    print("\nGenerating focused visualizations...")
    
    # Generate focused plots
    try:
        print("1. Algorithm performance comparison and thread scaling...")
        plot_algorithm_performance_comparison(df, csv_filename)
        
        print("2. Best algorithm heatmaps...")
        plot_best_algorithm_heatmap(df, csv_filename)
        
        print("3. Kernel size and matrix size impact analysis...")
        plot_kernel_and_size_impact(df, csv_filename)
        
        print("4. Generating focused performance report...")
        generate_focused_report(df)
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    base = os.path.splitext(csv_filename)[0]
    print("\n✅ Focused analysis complete! Generated files:")
    print(f"  - {base}/algorithm_performance_analysis.png")
    print(f"  - {base}/best_algorithm_heatmaps.png")
    print(f"  - {base}/kernel_and_size_impact.png")

if __name__ == "__main__":
    main()
