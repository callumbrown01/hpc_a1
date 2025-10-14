#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for 2D Convolution OpenMP Parallelization
Shows speedup vs kernel size, matrix size, thread count, and algorithm comparison
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
    # Read CSV, skipping comment lines
    df = pd.read_csv(filename, comment='#')
    
    # Remove rows with missing or invalid data
    df = df.dropna(subset=['Sequential_Time', 'Best_Time', 'Speedup'])
    df = df[df['Sequential_Time'] > 0]
    df = df[df['Best_Time'] > 0]
    df = df[df['Speedup'] > 0]
    
    # Calculate additional metrics
    if 'Efficiency' not in df.columns:
        df['Efficiency'] = df['Speedup'] / df['Threads']
    
    df['Problem_Size'] = df['Matrix_H'] * df['Matrix_W']
    df['Work_Per_Thread'] = df['Problem_Size'] / df['Threads']
    
    return df

def create_comprehensive_analysis(df, csv_filename):
    """Create comprehensive performance analysis figure"""
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Algorithm Performance Comparison (Top Left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        algo_perf = algo_data.groupby('Best_Algorithm')['Speedup'].mean().sort_values(ascending=False)
        bars = ax1.bar(range(len(algo_perf)), algo_perf.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'][:len(algo_perf)])
        ax1.set_xticks(range(len(algo_perf)))
        ax1.set_xticklabels([algo.replace('scheduling', '').replace('approach', '').strip() 
                            for algo in algo_perf.index], rotation=45, ha='right')
        ax1.set_ylabel('Average Speedup')
        ax1.set_title('Algorithm Performance Comparison', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, algo_perf.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 2. Speedup vs Kernel Size (Top Right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    kernel_speedups = df.groupby('Kernel_H')['Speedup'].agg(['mean', 'max']).reset_index()
    ax2.plot(kernel_speedups['Kernel_H'], kernel_speedups['mean'], 'bo-', 
             linewidth=3, markersize=8, label='Average Speedup')
    ax2.plot(kernel_speedups['Kernel_H'], kernel_speedups['max'], 'ro-', 
             linewidth=3, markersize=8, label='Maximum Speedup')
    ax2.set_xlabel('Kernel Size (NxN)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Kernel Size', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for peak performance
    for i, row in kernel_speedups.iterrows():
        ax2.annotate(f'{row["max"]:.1f}x', 
                    (row['Kernel_H'], row['max']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # 3. Speedup vs Matrix Size (Second Row Left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    matrix_speedups = df.groupby('Matrix_H')['Speedup'].agg(['mean', 'max']).reset_index()
    ax3.plot(matrix_speedups['Matrix_H'], matrix_speedups['mean'], 'go-', 
             linewidth=3, markersize=8, label='Average Speedup')
    ax3.plot(matrix_speedups['Matrix_H'], matrix_speedups['max'], 'mo-', 
             linewidth=3, markersize=8, label='Maximum Speedup')
    ax3.set_xlabel('Matrix Size (NxN)')
    ax3.set_ylabel('Speedup')
    ax3.set_title('Speedup vs Matrix Size', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Speedup vs Thread Count (Second Row Right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    thread_speedups = df.groupby('Threads')['Speedup'].agg(['mean', 'max']).reset_index()
    ax4.plot(thread_speedups['Threads'], thread_speedups['mean'], 'co-', 
             linewidth=3, markersize=8, label='Average Speedup')
    ax4.plot(thread_speedups['Threads'], thread_speedups['max'], 'yo-', 
             linewidth=3, markersize=8, label='Maximum Speedup')
    
    # Add ideal speedup line
    max_threads = thread_speedups['Threads'].max()
    ax4.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, linewidth=2, label='Ideal Speedup')
    
    ax4.set_xlabel('Number of Threads')
    ax4.set_ylabel('Speedup')
    ax4.set_title('Speedup vs Thread Count', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log', base=2)
    
    # 5. Algorithm Performance by Thread Count (Third Row)
    ax5 = fig.add_subplot(gs[2, :])
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        algorithms = ['Basic_Static_Time', 'Dynamic_Time', 'Guided_Time', 'Collapse_Time']
        algo_names = ['Basic Static', 'Dynamic', 'Guided', 'Collapse']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (algo_col, algo_name, color) in enumerate(zip(algorithms, algo_names, colors)):
            if algo_col in algo_data.columns:
                # Calculate speedup for this algorithm
                algo_data[f'{algo_name}_Speedup'] = algo_data['Sequential_Time'] / algo_data[algo_col]
                
                # Group by threads and get average speedup
                thread_speedup = algo_data.groupby('Threads')[f'{algo_name}_Speedup'].mean()
                
                ax5.plot(thread_speedup.index, thread_speedup.values, 'o-', 
                        linewidth=3, markersize=8, label=algo_name, color=color)
        
        # Add ideal speedup line
        max_threads = algo_data['Threads'].max()
        ax5.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, linewidth=2, label='Ideal Speedup')
        
        ax5.set_xlabel('Number of Threads')
        ax5.set_ylabel('Average Speedup')
        ax5.set_title('Algorithm Performance vs Thread Count', fontweight='bold', fontsize=14)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log', base=2)
        ax5.set_yscale('log', base=2)
    
    # 6. Best Algorithm Heatmap (Bottom Row)
    ax6 = fig.add_subplot(gs[3, :2])
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        try:
            # Create a pivot table for best algorithm by matrix size and kernel size
            pivot_algo = algo_data.pivot_table(
                index='Kernel_H', 
                columns='Matrix_H', 
                values='Best_Algorithm', 
                aggfunc=lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
            )
            
            if not pivot_algo.empty:
                # Create numerical mapping for algorithms
                unique_algos = algo_data['Best_Algorithm'].unique()
                algo_map = {algo: i for i, algo in enumerate(unique_algos)}
                
                # Convert to numerical values for plotting
                pivot_numeric = pivot_algo.map(algo_map)
                
                im = ax6.imshow(pivot_numeric.values, cmap='tab10', aspect='auto')
                ax6.set_xticks(range(len(pivot_algo.columns)))
                ax6.set_xticklabels([f'{size}' for size in pivot_algo.columns])
                ax6.set_yticks(range(len(pivot_algo.index)))
                ax6.set_yticklabels([f'{size}x{size}' for size in pivot_algo.index])
                ax6.set_xlabel('Matrix Size')
                ax6.set_ylabel('Kernel Size')
                ax6.set_title('Best Algorithm by Matrix and Kernel Size', fontweight='bold', fontsize=14)
                
                # Add text annotations
                for i in range(len(pivot_algo.index)):
                    for j in range(len(pivot_algo.columns)):
                        if not pd.isna(pivot_numeric.iloc[i, j]):
                            algo_name = pivot_algo.iloc[i, j]
                            short_name = algo_name.replace('scheduling', '').replace('approach', '').strip()
                            ax6.text(j, i, short_name, ha="center", va="center", 
                                   color='white', fontweight='bold', fontsize=10)
        except Exception as e:
            ax6.text(0.5, 0.5, 'Best Algorithm data unavailable', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Best Algorithm by Matrix and Kernel Size')
    
    # 7. Efficiency Analysis (Bottom Right)
    ax7 = fig.add_subplot(gs[3, 2:])
    efficiency_by_threads = df.groupby('Threads')['Efficiency'].mean() * 100
    bars = ax7.bar(range(len(efficiency_by_threads)), efficiency_by_threads.values, 
                   color='purple', alpha=0.7)
    ax7.set_xticks(range(len(efficiency_by_threads)))
    ax7.set_xticklabels(efficiency_by_threads.index)
    ax7.set_ylabel('Average Efficiency (%)')
    ax7.set_xlabel('Number of Threads')
    ax7.set_title('Parallel Efficiency by Thread Count', fontweight='bold', fontsize=14)
    ax7.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Perfect Efficiency')
    ax7.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Efficiency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency_by_threads.values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add overall title
    fig.suptitle('OpenMP 2D Convolution Performance Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save the figure
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'comprehensive_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_recommendations(df):
    """Generate performance recommendations based on analysis"""
    print("\n" + "="*80)
    print("🎯 PERFORMANCE RECOMMENDATIONS")
    print("="*80)
    
    # Algorithm recommendations
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        print("\n📊 ALGORITHM RECOMMENDATIONS:")
        
        # Best overall algorithm
        best_overall = algo_data.groupby('Best_Algorithm')['Speedup'].mean().idxmax()
        best_speedup = algo_data.groupby('Best_Algorithm')['Speedup'].mean().max()
        print(f"• Best Overall Algorithm: {best_overall} (avg {best_speedup:.2f}x speedup)")
        
        # Best for different thread counts
        for threads in [1, 4, 8, 16, 32, 48]:
            thread_data = algo_data[algo_data['Threads'] == threads]
            if not thread_data.empty:
                best_for_threads = thread_data.groupby('Best_Algorithm')['Speedup'].mean().idxmax()
                speedup = thread_data.groupby('Best_Algorithm')['Speedup'].mean().max()
                print(f"• Best for {threads} threads: {best_for_threads} ({speedup:.2f}x speedup)")
    
    # Matrix size recommendations
    print("\n📐 MATRIX SIZE INSIGHTS:")
    matrix_perf = df.groupby('Matrix_H')['Speedup'].agg(['mean', 'max']).reset_index()
    best_matrix = matrix_perf.loc[matrix_perf['max'].idxmax()]
    print(f"• Best matrix size for peak performance: {int(best_matrix['Matrix_H'])}x{int(best_matrix['Matrix_H'])} ({best_matrix['max']:.2f}x speedup)")
    
    # Kernel size recommendations
    print("\n🔍 KERNEL SIZE INSIGHTS:")
    kernel_perf = df.groupby('Kernel_H')['Speedup'].agg(['mean', 'max']).reset_index()
    best_kernel = kernel_perf.loc[kernel_perf['max'].idxmax()]
    print(f"• Best kernel size for peak performance: {int(best_kernel['Kernel_H'])}x{int(best_kernel['Kernel_H'])} ({best_kernel['max']:.2f}x speedup)")
    
    # Thread scaling recommendations
    print("\n🧵 THREAD SCALING RECOMMENDATIONS:")
    thread_eff = df.groupby('Threads')['Efficiency'].mean()
    
    # Find optimal thread count (best efficiency above 70%)
    good_efficiency = thread_eff[thread_eff > 0.7]
    if not good_efficiency.empty:
        optimal_threads = good_efficiency.index.max()
        print(f"• Optimal thread count for efficiency: {optimal_threads} threads ({thread_eff[optimal_threads]*100:.1f}% efficiency)")
    
    # Find maximum speedup thread count
    max_speedup_threads = df.groupby('Threads')['Speedup'].max().idxmax()
    max_speedup_value = df.groupby('Threads')['Speedup'].max().max()
    print(f"• Thread count for maximum speedup: {max_speedup_threads} threads ({max_speedup_value:.2f}x speedup)")
    
    # Scalability analysis
    print("\n📈 SCALABILITY ANALYSIS:")
    efficiency_32 = thread_eff.get(32, 0) * 100
    efficiency_48 = thread_eff.get(48, 0) * 100
    if efficiency_32 > 0:
        print(f"• Efficiency at 32 threads: {efficiency_32:.1f}%")
    if efficiency_48 > 0:
        print(f"• Efficiency at 48 threads: {efficiency_48:.1f}%")
    
    if efficiency_32 > 50:
        print("✅ Good scalability up to 32 threads")
    elif efficiency_32 > 30:
        print("⚠️  Moderate scalability at 32 threads")
    else:
        print("❌ Poor scalability beyond 16 threads")
    
    print("\n" + "="*80)

def main():
    """Main function"""
    print("🔬 OpenMP 2D Convolution Performance Analysis")
    print("="*60)
    
    csv_filename = 'performance_test.csv'
    try:
        df = load_and_clean_data(csv_filename)
        print(f"✅ Loaded {len(df)} data points from {csv_filename}")
        print(f"📊 Test categories: {list(df['Test_Category'].unique())}")
        
        print("\n🎨 Creating comprehensive performance analysis...")
        create_comprehensive_analysis(df, csv_filename)
        
        generate_recommendations(df)
        
        base = os.path.splitext(csv_filename)[0]
        print(f"\n📁 Analysis saved to: {base}/comprehensive_performance_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
