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
    df = df.dropna(subset=['Sequential_Time', 'Best_Time', 'Speedup'])
    df = df[df['Sequential_Time'] > 0]
    df = df[df['Best_Time'] > 0]
    df = df[df['Speedup'] > 0]
    
    # Calculate additional metrics
    df['Efficiency'] = df['Speedup'] / df['Threads']
    df['Problem_Size'] = df['Matrix_H'] * df['Matrix_W']
    df['Work_Per_Thread'] = df['Problem_Size'] / df['Threads']
    df['Parallel_Time'] = df['Best_Time']  # Use best time as parallel time
    
    return df

def plot_thread_scaling(df, csv_filename):
    """Plot thread scaling analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter thread scaling data
    thread_data = df[df['Test_Category'] == 'Thread_Scalability'].copy()
    
    # Group by problem size
    sizes = thread_data['Problem_Size'].unique()
    sizes = sorted(sizes)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    # 1. Speedup vs Threads
    for i, size in enumerate(sizes):
        data = thread_data[thread_data['Problem_Size'] == size]
        if not data.empty:
            # Get problem description for legend
            problem_desc = data['Problem_Size'].iloc[0]
            matrix_size = int(np.sqrt(problem_desc))
            ax1.plot(data['Threads'], data['Speedup'], 'o-', 
                    label=f'{matrix_size}×{matrix_size}', 
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
        if not data.empty:
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
        if not data.empty:
            matrix_size = int(np.sqrt(size))
            ax3.plot(data['Threads'], data['Sequential_Time'], '--', 
                    color=colors[i], alpha=0.7, label=f'Sequential ({matrix_size}×{matrix_size})')
            ax3.plot(data['Threads'], data['Parallel_Time'], 'o-', 
                    color=colors[i], linewidth=2, markersize=6, 
                    label=f'Parallel ({matrix_size}×{matrix_size})')
    
    ax3.set_xlabel('Number of Threads')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Thread Scaling: Execution Time')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    
    # 4. Speedup Heatmap
    if not thread_data.empty and 'Problem_Size' in thread_data.columns:
        pivot_data = thread_data.pivot_table(index='Problem_Size', columns='Threads', values='Speedup', aggfunc='mean')
        if not pivot_data.empty:
            im = ax4.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(len(pivot_data.columns)))
            ax4.set_xticklabels(pivot_data.columns)
            ax4.set_yticks(range(len(pivot_data.index)))
            ax4.set_yticklabels([f'{int(np.sqrt(size))}²' for size in pivot_data.index])
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
    """Plot algorithm comparison and kernel impact analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Algorithm comparison data
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison'].copy()
    
    # 1. Algorithm Performance Comparison
    if not algo_data.empty:
        # Group by algorithm type
        algo_perf = algo_data.groupby('Best_Algorithm')['Speedup'].mean().sort_values(ascending=False)
        
        bars = ax1.bar(range(len(algo_perf)), algo_perf.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'][:len(algo_perf)])
        ax1.set_xticks(range(len(algo_perf)))
        ax1.set_xticklabels(algo_perf.index, rotation=45)
        ax1.set_ylabel('Average Speedup')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, algo_perf.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # 2. Kernel Impact Analysis
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
        
        ax2.plot(kernel_sizes, speedups, 'go-', linewidth=3, markersize=8, label='Best Speedup')
        ax2.set_xlabel('Kernel Size')
        ax2.set_ylabel('Maximum Speedup Achieved')
        ax2.set_title('Kernel Size Impact on Performance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add trend line
        if len(kernel_sizes) > 1:
            z = np.polyfit(kernel_sizes, speedups, 1)
            p = np.poly1d(z)
            ax2.plot(kernel_sizes, p(kernel_sizes), "r--", alpha=0.8)
    
    # 3. Cache Performance Analysis
    cache_data = df[df['Test_Category'].isin(['Cache_Friendly', 'Cache_Unfriendly'])].copy()
    
    if not cache_data.empty:
        friendly = cache_data[cache_data['Test_Category'] == 'Cache_Friendly']
        unfriendly = cache_data[cache_data['Test_Category'] == 'Cache_Unfriendly']
        
        if not friendly.empty and not unfriendly.empty:
            # Average efficiency by thread count
            friendly_eff = friendly.groupby('Threads')['Efficiency'].mean() * 100
            unfriendly_eff = unfriendly.groupby('Threads')['Efficiency'].mean() * 100
            
            ax3.plot(friendly_eff.index, friendly_eff.values, 'b-o', 
                    linewidth=2, markersize=6, label='Cache-Friendly')
            ax3.plot(unfriendly_eff.index, unfriendly_eff.values, 'r-s', 
                    linewidth=2, markersize=6, label='Cache-Unfriendly')
            
            ax3.set_xlabel('Number of Threads')
            ax3.set_ylabel('Average Efficiency (%)')
            ax3.set_title('Cache Impact on Parallel Efficiency')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            ax3.set_xscale('log', base=2)
    
    # 4. Thread Scalability Summary
    thread_data = df[df['Test_Category'] == 'Thread_Scalability'].copy()
    
    if not thread_data.empty:
        # Efficiency vs Work per Thread
        scatter = ax4.scatter(thread_data['Work_Per_Thread'], thread_data['Efficiency'] * 100, 
                             c=thread_data['Threads'], s=60, alpha=0.7, cmap='plasma')
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
    
    # 1. Overall Performance Summary by Category
    categories = df['Test_Category'].unique()
    max_speedups = []
    avg_speedups = []
    best_efficiencies = []
    
    for category in categories:
        cat_data = df[df['Test_Category'] == category]
        max_speedups.append(cat_data['Speedup'].max())
        avg_speedups.append(cat_data['Speedup'].mean())
        best_efficiencies.append(cat_data['Efficiency'].max() * 100)
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax1.bar(x - width, max_speedups, width, label='Max Speedup', alpha=0.8, color='lightblue')
    ax1.bar(x, avg_speedups, width, label='Avg Speedup', alpha=0.8, color='lightgreen')
    ax1.bar(x + width, [eff/10 for eff in best_efficiencies], width, 
            label='Best Efficiency/10', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Test Category')
    ax1.set_ylabel('Performance Metrics')
    ax1.set_title('Performance Summary by Test Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Algorithm Efficiency Comparison
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        algo_efficiency = algo_data.groupby('Best_Algorithm')['Efficiency'].mean() * 100
        
        wedges, texts, autotexts = ax2.pie(algo_efficiency.values, labels=algo_efficiency.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Algorithm Efficiency Distribution')
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
    
    # 3. Speedup vs Problem Size (Thread Scalability)
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    if not thread_data.empty:
        # Group by problem size and show best speedup achieved
        problem_speedups = thread_data.groupby('Problem_Size')['Speedup'].max()
        problem_sizes = [int(np.sqrt(size)) for size in problem_speedups.index]
        
        ax3.plot(problem_sizes, problem_speedups.values, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Matrix Size (N×N)')
        ax3.set_ylabel('Maximum Speedup Achieved')
        ax3.set_title('Peak Performance vs Problem Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Annotate peak performance points
        for size, speedup in zip(problem_sizes, problem_speedups.values):
            ax3.annotate(f'{speedup:.1f}x', (size, speedup), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Thread Efficiency Heatmap
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    if not thread_data.empty and len(thread_data) > 1:
        try:
            # Create efficiency heatmap
            pivot_eff = thread_data.pivot_table(
                index='Problem_Size', 
                columns='Threads', 
                values='Efficiency', 
                aggfunc='mean'
            )
            
            if not pivot_eff.empty:
                im = ax4.imshow(pivot_eff.values * 100, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
                ax4.set_xticks(range(len(pivot_eff.columns)))
                ax4.set_xticklabels(pivot_eff.columns)
                ax4.set_yticks(range(len(pivot_eff.index)))
                ax4.set_yticklabels([f'{int(np.sqrt(size))}²' for size in pivot_eff.index])
                ax4.set_xlabel('Number of Threads')
                ax4.set_ylabel('Matrix Size')
                ax4.set_title('Parallel Efficiency Heatmap (%)')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Efficiency (%)')
                
                # Add text annotations
                for i in range(len(pivot_eff.index)):
                    for j in range(len(pivot_eff.columns)):
                        if not pd.isna(pivot_eff.iloc[i, j]):
                            eff_val = pivot_eff.iloc[i, j] * 100
                            color = 'white' if eff_val < 50 else 'black'
                            ax4.text(j, i, f'{eff_val:.0f}', 
                                   ha="center", va="center", color=color, fontweight='bold')
        except Exception as e:
            ax4.text(0.5, 0.5, 'Heatmap data unavailable', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Parallel Efficiency Heatmap')
    
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
    best_overall = df.groupby('Test_Category')['Speedup'].max()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'][:len(best_overall)]
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
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    
    if not thread_data.empty:
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
    
    # 3. Algorithm performance comparison
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    if not algo_data.empty:
        algo_performance = algo_data.groupby('Best_Algorithm')['Speedup'].mean().sort_values(ascending=True)
        
        bars3 = ax3.barh(range(len(algo_performance)), algo_performance.values, 
                        color=['lightblue', 'lightgreen', 'lightcoral', 'gold', 'orange'][:len(algo_performance)])
        ax3.set_yticks(range(len(algo_performance)))
        ax3.set_yticklabels(algo_performance.index)
        ax3.set_xlabel('Average Speedup')
        ax3.set_title('Algorithm Performance Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars3, algo_performance.values)):
            ax3.text(value + 0.1, i, f'{value:.1f}x', 
                    va='center', ha='left', fontweight='bold')
    
    # 4. Cache performance impact
    cache_friendly = df[df['Test_Category'] == 'Cache_Friendly']
    cache_unfriendly = df[df['Test_Category'] == 'Cache_Unfriendly']
    
    if not cache_friendly.empty and not cache_unfriendly.empty:
        friendly_eff = cache_friendly.groupby('Threads')['Efficiency'].mean() * 100
        unfriendly_eff = cache_unfriendly.groupby('Threads')['Efficiency'].mean() * 100
        
        common_threads = sorted(set(friendly_eff.index) & set(unfriendly_eff.index))
        if common_threads:
            friendly_vals = [friendly_eff.get(t, 0) for t in common_threads]
            unfriendly_vals = [unfriendly_eff.get(t, 0) for t in common_threads]
            
            ax4.plot(common_threads, friendly_vals, 'b-o', linewidth=2, markersize=6, label='Cache-Friendly')
            ax4.plot(common_threads, unfriendly_vals, 'r-s', linewidth=2, markersize=6, label='Cache-Unfriendly')
            ax4.set_xlabel('Number of Threads')
            ax4.set_ylabel('Average Efficiency (%)')
            ax4.set_title('Cache Impact on Performance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_xscale('log', base=2)
    
    # 5. Performance overview across all categories
    categories = df['Test_Category'].unique()
    perf_summary = []
    
    for category in categories:
        cat_data = df[df['Test_Category'] == category]
        if not cat_data.empty:
            perf_summary.append({
                'Category': category,
                'Max_Speedup': cat_data['Speedup'].max(),
                'Avg_Speedup': cat_data['Speedup'].mean(),
                'Best_Efficiency': cat_data['Efficiency'].max() * 100,
                'Configurations': len(cat_data)
            })
    
    if perf_summary:
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
        ax5.set_xticklabels(summary_df['Category'], rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
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
    thread_data = df[df['Test_Category'] == 'Thread_Scalability']
    print(f"\n🧵 THREAD SCALING INSIGHTS")
    if not thread_data.empty:
        print(f"Best thread scaling speedup: {thread_data['Speedup'].max():.2f}x")
        
        # Find optimal thread configuration
        best_thread_config = thread_data.loc[thread_data['Speedup'].idxmax()]
        if 'Problem_Size' in best_thread_config:
            matrix_size = int(np.sqrt(best_thread_config['Problem_Size']))
            print(f"Optimal configuration: {matrix_size}×{matrix_size} matrix with {best_thread_config['Threads']} threads")
        
        # Efficiency analysis
        high_eff = thread_data[thread_data['Efficiency'] > 0.5]
        print(f"Configurations with >50% efficiency: {len(high_eff)}/{len(thread_data)} ({len(high_eff)/len(thread_data)*100:.1f}%)")
    else:
        print("No thread scaling data available.")
    
    # Algorithm comparison insights
    algo_data = df[df['Test_Category'] == 'Algorithm_Comparison']
    print(f"\n� ALGORITHM COMPARISON INSIGHTS")
    if not algo_data.empty:
        best_algo = algo_data.groupby('Best_Algorithm')['Speedup'].mean().idxmax()
        best_algo_speedup = algo_data.groupby('Best_Algorithm')['Speedup'].mean().max()
        print(f"Best performing algorithm: {best_algo} (avg {best_algo_speedup:.2f}x speedup)")
        
        algo_counts = algo_data['Best_Algorithm'].value_counts()
        print("Algorithm usage frequency:")
        for algo, count in algo_counts.items():
            print(f"  - {algo}: {count} times ({count/len(algo_data)*100:.1f}%)")
    else:
        print("No algorithm comparison data available.")
    
    # Kernel size insights
    kernel_data = df[df['Test_Category'] == 'Kernel_Impact']
    print(f"\n🔍 KERNEL SIZE INSIGHTS")
    if not kernel_data.empty and kernel_data['Speedup'].notna().any():
        best_idx = kernel_data['Speedup'].idxmax()
        if best_idx in kernel_data.index:
            best_kernel_size = kernel_data.loc[best_idx, 'Kernel_H']
            best_speedup = kernel_data.loc[best_idx, 'Speedup']
            print(f"Best kernel size performance: {best_kernel_size}×{best_kernel_size} kernel ({best_speedup:.2f}x speedup)")
        else:
            print("No valid kernel scaling data available.")
    else:
        print("No kernel scaling data available.")
    
    # Cache performance insights
    cache_friendly = df[df['Test_Category'] == 'Cache_Friendly']
    cache_unfriendly = df[df['Test_Category'] == 'Cache_Unfriendly']
    print(f"\n🧮 CACHE PERFORMANCE INSIGHTS")
    if not cache_friendly.empty and not cache_unfriendly.empty:
        friendly_avg_eff = cache_friendly['Efficiency'].mean() * 100
        unfriendly_avg_eff = cache_unfriendly['Efficiency'].mean() * 100
        print(f"Cache-friendly average efficiency: {friendly_avg_eff:.1f}%")
        print(f"Cache-unfriendly average efficiency: {unfriendly_avg_eff:.1f}%")
        print(f"Cache impact: {friendly_avg_eff - unfriendly_avg_eff:.1f}% efficiency difference")
    else:
        print("No cache performance comparison data available.")
    
    # Scalability analysis
    print(f"\n📈 SCALABILITY ANALYSIS")
    if 'Problem_Size' in thread_data.columns and not thread_data.empty:
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
    filenames = ['performance_test.csv']
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
