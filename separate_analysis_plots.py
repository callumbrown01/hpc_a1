#!/usr/bin/env python3
"""
Separate High-Resolution Analysis Plots for HPC Report
Creates 6 individual PNG files:
1. Best Algorithm Selection Matrix
2. Guided Algorithm Speedup vs Thread Count by Matrix Size
3a. All Algorithms Speedup vs Problem Size at 8 Threads
3b. All Algorithms Speedup vs Problem Size at 16 Threads
3c. All Algorithms Speedup vs Problem Size at 96 Threads
4. Average Performance Heatmap (Problem Size vs Thread Count)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for high-quality output
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(filename):
    """Load CSV data and clean it"""
    df = pd.read_csv(filename, comment='#')
    
    # Remove rows with missing or invalid data
    df = df.dropna(subset=['Sequential_Time', 'Best_Time', 'Speedup', 'Best_Algorithm'])
    df = df[df['Sequential_Time'] > 0]
    df = df[df['Best_Time'] > 0]
    df = df[df['Speedup'] > 0]
    
    # Filter to only include 4+ threads since single thread data was removed
    df = df[df['Threads'] >= 4]
    
    # Calculate problem size
    df['Problem_Size'] = df['Matrix_H'] * df['Matrix_W']
    
    return df

def plot1_algorithm_selection_matrix(df, output_dir):
    """Plot 1: Best Algorithm Selection Matrix"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create problem size categories
    df['Problem_Size_Category'] = pd.cut(df['Problem_Size'], 
                                        bins=[0, 100000, 1000000, 10000000, 100000000, float('inf')],
                                        labels=['Small\n(<100K)', 'Medium\n(100K-1M)', 'Large\n(1M-10M)', 
                                               'Very Large\n(10M-100M)', 'Huge\n(>100M)'])
    
    # Create pivot table for best algorithm
    pivot_algo = df.pivot_table(
        index='Problem_Size_Category', 
        columns='Threads', 
        values='Best_Algorithm', 
        aggfunc=lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
    )
    
    if not pivot_algo.empty:
        # Create numerical mapping for algorithms
        algo_colors_map = {
            'Basicstatic': 0,
            'Dynamicscheduling': 1, 
            'Guidedscheduling': 2,
            'Collapseapproach': 3
        }
        
        # Convert to numerical values for plotting
        def safe_map(x):
            if pd.isna(x) or not isinstance(x, str):
                return -1  # Use -1 for missing values
            return algo_colors_map.get(x, -1)
        
        pivot_numeric = pivot_algo.applymap(safe_map)
        
        # Create custom colormap
        colors_heat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors_heat[:len(algo_colors_map)])
        
        im = ax.imshow(pivot_numeric.values, cmap=cmap, aspect='auto', vmin=0, vmax=3)
        
        ax.set_xticks(range(len(pivot_algo.columns)))
        ax.set_xticklabels(pivot_algo.columns, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(pivot_algo.index)))
        ax.set_yticklabels(pivot_algo.index, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Threads', fontsize=16, fontweight='bold')
        ax.set_ylabel('Problem Size Category', fontsize=16, fontweight='bold')
        ax.set_title('Best Algorithm Selection Matrix\nOptimal Algorithm by Problem Size & Thread Count', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(len(pivot_algo.index)):
            for j in range(len(pivot_algo.columns)):
                if not pd.isna(pivot_numeric.iloc[i, j]):
                    algo_name = pivot_algo.iloc[i, j]
                    # Handle NaN values properly
                    if pd.isna(algo_name) or not isinstance(algo_name, str):
                        short_name = "N/A"
                    else:
                        short_name = algo_name.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                    ax.text(j, i, short_name, ha="center", va="center", 
                           color='white', fontweight='bold', fontsize=12)
        
        # Create legend
        legend_elements = []
        for algo, color_idx in algo_colors_map.items():
            if any(algo in unique_algo for unique_algo in df['Best_Algorithm'].unique()):
                clean_name = algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors_heat[color_idx], label=clean_name))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_algorithm_selection_matrix.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 1_algorithm_selection_matrix.png")

def plot2_speedup_vs_threads(df, output_dir):
    """Plot 2: Guided Algorithm Speedup vs Thread Count by Matrix Size (Grouped Bar Chart)"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Focus on Guided algorithm only
    if 'Guided_Time' in df.columns:
        df['Guided_Speedup'] = df['Sequential_Time'] / df['Guided_Time']
        
        # Get unique matrix sizes and thread counts
        matrix_sizes = sorted(df['Problem_Size'].unique())
        thread_counts = sorted(df['Threads'].unique())
        
        # Create a pivot table for easier plotting
        pivot_data = df.pivot_table(
            index='Threads', 
            columns='Problem_Size', 
            values='Guided_Speedup', 
            aggfunc='mean'
        )
        
        # Prepare data for grouped bar chart
        x = np.arange(len(thread_counts))  # Thread count positions
        width = 0.1  # Width of each bar
        colors = plt.cm.viridis(np.linspace(0, 1, len(matrix_sizes)))
        
        # Create bars for each matrix size
        for i, matrix_size in enumerate(matrix_sizes):
            if matrix_size in pivot_data.columns:
                speedups = [pivot_data.loc[tc, matrix_size] if not pd.isna(pivot_data.loc[tc, matrix_size]) else 0 
                           for tc in thread_counts]
                
                # Format matrix size label
                if matrix_size >= 1000000:
                    size_label = f'{matrix_size/1000000:.0f}M'
                elif matrix_size >= 1000:
                    size_label = f'{matrix_size/1000:.0f}K'
                else:
                    size_label = f'{matrix_size:.0f}'
                
                # Plot bars with offset
                offset = (i - len(matrix_sizes)/2) * width
                bars = ax.bar(x + offset, speedups, width, 
                             label=f'{size_label} elements', 
                             color=colors[i], alpha=0.8)
                
                # Add value labels on top of bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}', ha='center', va='bottom', 
                               fontsize=8, rotation=90)
        
        # Customize the plot
        ax.set_xlabel('Number of Threads', fontsize=16, fontweight='bold')
        ax.set_ylabel('Speedup (Guided Algorithm)', fontsize=16, fontweight='bold')
        ax.set_title('Guided Algorithm: Speedup by Thread Count and Matrix Size\nGrouped Bar Chart Comparison', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels([str(tc) for tc in thread_counts], fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Create legend with multiple columns
        ax.legend(fontsize=10, loc='upper left', ncol=3, frameon=True, fancybox=True, shadow=True)
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Set y-axis to start from 0 for better bar chart visualization
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_speedup_vs_threads.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 2_speedup_vs_threads.png")

def plot3a_algorithms_vs_problem_size_8threads(df, output_dir):
    """Plot 3a: All Algorithms Speedup vs Problem Size at 8 Threads"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter data for 8 threads only
    df_8 = df[df['Threads'] == 8].copy()
    
    if not df_8.empty:
        algorithms = {
            'Basic_Static_Time': 'Basic Static',
            'Dynamic_Time': 'Dynamic', 
            'Guided_Time': 'Guided',
            'Collapse_Time': 'Collapse'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        markers = ['o', 's', '^', 'D']
        
        # Calculate speedups for each algorithm
        for algo_col, algo_name in algorithms.items():
            if algo_col in df_8.columns:
                df_8[f'{algo_name}_Speedup'] = df_8['Sequential_Time'] / df_8[algo_col]
        
        # Plot each algorithm
        for i, (algo_col, algo_name) in enumerate(algorithms.items()):
            if f'{algo_name}_Speedup' in df_8.columns:
                # Group by problem size and get mean speedup
                size_stats = df_8.groupby('Problem_Size')[f'{algo_name}_Speedup'].mean().reset_index()
                
                ax.plot(size_stats['Problem_Size'], size_stats[f'{algo_name}_Speedup'], 
                       f'{markers[i]}-', linewidth=3, markersize=8,
                       color=colors[i], label=f'{algo_name}', alpha=0.8)
        
        # Add sequential baseline (speedup = 1)
        problem_sizes = sorted(df_8['Problem_Size'].unique())
        sequential_speedup = [1.0] * len(problem_sizes)
        ax.plot(problem_sizes, sequential_speedup, 'x-', 
               linewidth=2, markersize=8, color='#9B59B6', 
               label='Sequential Baseline', alpha=0.8)
        
        ax.set_xlabel('Problem Size (Matrix Elements)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Speedup at 8 Threads', fontsize=16, fontweight='bold')
        ax.set_title('Algorithm Performance vs Problem Size at 8 Threads\nSpeedup Comparison Across Different Matrix Sizes', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xscale('log', base=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3a_algorithms_vs_problem_size_8threads.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 3a_algorithms_vs_problem_size_8threads.png")

def plot3b_algorithms_vs_problem_size_16threads(df, output_dir):
    """Plot 3b: All Algorithms Speedup vs Problem Size at 16 Threads"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter data for 16 threads only
    df_16 = df[df['Threads'] == 16].copy()
    
    if not df_16.empty:
        algorithms = {
            'Basic_Static_Time': 'Basic Static',
            'Dynamic_Time': 'Dynamic', 
            'Guided_Time': 'Guided',
            'Collapse_Time': 'Collapse'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        markers = ['o', 's', '^', 'D']
        
        # Calculate speedups for each algorithm
        for algo_col, algo_name in algorithms.items():
            if algo_col in df_16.columns:
                df_16[f'{algo_name}_Speedup'] = df_16['Sequential_Time'] / df_16[algo_col]
        
        # Plot each algorithm
        for i, (algo_col, algo_name) in enumerate(algorithms.items()):
            if f'{algo_name}_Speedup' in df_16.columns:
                # Group by problem size and get mean speedup
                size_stats = df_16.groupby('Problem_Size')[f'{algo_name}_Speedup'].mean().reset_index()
                
                ax.plot(size_stats['Problem_Size'], size_stats[f'{algo_name}_Speedup'], 
                       f'{markers[i]}-', linewidth=3, markersize=8,
                       color=colors[i], label=f'{algo_name}', alpha=0.8)
        
        # Add sequential baseline (speedup = 1)
        problem_sizes = sorted(df_16['Problem_Size'].unique())
        sequential_speedup = [1.0] * len(problem_sizes)
        ax.plot(problem_sizes, sequential_speedup, 'x-', 
               linewidth=2, markersize=8, color='#9B59B6', 
               label='Sequential Baseline', alpha=0.8)
        
        ax.set_xlabel('Problem Size (Matrix Elements)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Speedup at 16 Threads', fontsize=16, fontweight='bold')
        ax.set_title('Algorithm Performance vs Problem Size at 16 Threads\nSpeedup Comparison Across Different Matrix Sizes', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xscale('log', base=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3b_algorithms_vs_problem_size_16threads.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 3b_algorithms_vs_problem_size_16threads.png")

def plot3c_algorithms_vs_problem_size_96threads(df, output_dir):
    """Plot 3c: All Algorithms Speedup vs Problem Size at 96 Threads"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter data for 96 threads only
    df_96 = df[df['Threads'] == 96].copy()
    
    if not df_96.empty:
        algorithms = {
            'Basic_Static_Time': 'Basic Static',
            'Dynamic_Time': 'Dynamic', 
            'Guided_Time': 'Guided',
            'Collapse_Time': 'Collapse'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        markers = ['o', 's', '^', 'D']
        
        # Calculate speedups for each algorithm
        for algo_col, algo_name in algorithms.items():
            if algo_col in df_96.columns:
                df_96[f'{algo_name}_Speedup'] = df_96['Sequential_Time'] / df_96[algo_col]
        
        # Plot each algorithm
        for i, (algo_col, algo_name) in enumerate(algorithms.items()):
            if f'{algo_name}_Speedup' in df_96.columns:
                # Group by problem size and get mean speedup
                size_stats = df_96.groupby('Problem_Size')[f'{algo_name}_Speedup'].mean().reset_index()
                
                ax.plot(size_stats['Problem_Size'], size_stats[f'{algo_name}_Speedup'], 
                       f'{markers[i]}-', linewidth=3, markersize=8,
                       color=colors[i], label=f'{algo_name}', alpha=0.8)
        
        # Add sequential baseline (speedup = 1)
        problem_sizes = sorted(df_96['Problem_Size'].unique())
        sequential_speedup = [1.0] * len(problem_sizes)
        ax.plot(problem_sizes, sequential_speedup, 'x-', 
               linewidth=2, markersize=8, color='#9B59B6', 
               label='Sequential Baseline', alpha=0.8)
        
        ax.set_xlabel('Problem Size (Matrix Elements)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Speedup at 96 Threads', fontsize=16, fontweight='bold')
        ax.set_title('Algorithm Performance vs Problem Size at 96 Threads\nSpeedup Comparison Across Different Matrix Sizes', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xscale('log', base=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper left')
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3c_algorithms_vs_problem_size_96threads.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 3c_algorithms_vs_problem_size_96threads.png")

def plot4_performance_heatmap(df, output_dir):
    """Plot 4: Average Performance Heatmap (Problem Size vs Thread Count)"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create problem size categories
    df['Problem_Size_Category'] = pd.cut(df['Problem_Size'], 
                                        bins=[0, 100000, 1000000, 10000000, 100000000, float('inf')],
                                        labels=['Small\n(<100K)', 'Medium\n(100K-1M)', 'Large\n(1M-10M)', 
                                               'Very Large\n(10M-100M)', 'Huge\n(>100M)'])
    
    # Create pivot table for average speedup
    pivot_speedup = df.pivot_table(
        index='Problem_Size_Category', 
        columns='Threads', 
        values='Speedup', 
        aggfunc='mean'
    )
    
    if not pivot_speedup.empty:
        # Create heatmap
        im = ax.imshow(pivot_speedup.values, cmap='RdYlBu_r', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Speedup', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        ax.set_xticks(range(len(pivot_speedup.columns)))
        ax.set_xticklabels(pivot_speedup.columns, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(pivot_speedup.index)))
        ax.set_yticklabels(pivot_speedup.index, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Threads', fontsize=16, fontweight='bold')
        ax.set_ylabel('Problem Size Category', fontsize=16, fontweight='bold')
        ax.set_title('Average Performance Heatmap\nSpeedup by Problem Size & Thread Count', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(len(pivot_speedup.index)):
            for j in range(len(pivot_speedup.columns)):
                if not pd.isna(pivot_speedup.iloc[i, j]):
                    speedup_val = pivot_speedup.iloc[i, j]
                    ax.text(j, i, f'{speedup_val:.1f}x', ha="center", va="center", 
                           color='white', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_performance_heatmap.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Created: 4_performance_heatmap.png")

def create_comprehensive_analysis(df, output_dir):
    """Create comprehensive analysis for each figure"""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE FIGURE ANALYSIS")
    print("="*100)
    
    # Calculate speedups for all algorithms
    algorithms = {
        'Basic_Static_Time': 'Basic Static',
        'Dynamic_Time': 'Dynamic', 
        'Guided_Time': 'Guided',
        'Collapse_Time': 'Collapse'
    }
    
    for algo_col, algo_name in algorithms.items():
        if algo_col in df.columns:
            df[f'{algo_name}_Speedup'] = df['Sequential_Time'] / df[algo_col]
    
    print("\n🔍 FIGURE 1: ALGORITHM SELECTION MATRIX ANALYSIS")
    print("-" * 60)
    
    # Create problem size categories for analysis
    df['Problem_Size_Category'] = pd.cut(df['Problem_Size'], 
                                        bins=[0, 100000, 1000000, 10000000, 100000000, float('inf')],
                                        labels=['Small', 'Medium', 'Large', 'Very Large', 'Huge'])
    
    # Analyze best algorithm distribution
    best_algo_counts = df['Best_Algorithm'].value_counts()
    print("📈 Algorithm Selection Distribution:")
    for algo, count in best_algo_counts.items():
        percentage = (count / len(df)) * 100
        clean_name = algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
        print(f"  • {clean_name}: {count} cases ({percentage:.1f}%)")
    
    # Analysis by thread count
    print("\n🧵 Best Algorithm by Thread Count:")
    for threads in sorted(df['Threads'].unique()):
        thread_data = df[df['Threads'] == threads]
        best_algo = thread_data['Best_Algorithm'].mode().iloc[0] if not thread_data.empty else "None"
        clean_name = best_algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
        avg_speedup = thread_data['Speedup'].mean()
        print(f"  • {threads:>2} threads: {clean_name:<12} (avg {avg_speedup:.1f}x speedup)")
    
    # Analysis by problem size
    print("\n📐 Best Algorithm by Problem Size Category:")
    for category in ['Small', 'Medium', 'Large', 'Very Large', 'Huge']:
        cat_data = df[df['Problem_Size_Category'] == category]
        if not cat_data.empty:
            best_algo = cat_data['Best_Algorithm'].mode().iloc[0]
            clean_name = best_algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
            avg_speedup = cat_data['Speedup'].mean()
            print(f"  • {category:<12}: {clean_name:<12} (avg {avg_speedup:.1f}x speedup)")
    
    print("\n🔍 FIGURE 2: GUIDED ALGORITHM BAR CHART ANALYSIS")
    print("-" * 60)
    
    if 'Guided_Speedup' in df.columns:
        print("📊 Guided Algorithm Performance Trends:")
        
        # Performance by thread count
        print("\n🧵 Speedup by Thread Count (Average across all matrix sizes):")
        for threads in sorted(df['Threads'].unique()):
            thread_data = df[df['Threads'] == threads]
            avg_speedup = thread_data['Guided_Speedup'].mean()
            efficiency = (avg_speedup / threads) * 100
            print(f"  • {threads:>2} threads: {avg_speedup:>6.1f}x speedup ({efficiency:>5.1f}% efficiency)")
        
        # Performance by matrix size
        print("\n📐 Speedup by Matrix Size (Average across all thread counts):")
        matrix_sizes = sorted(df['Problem_Size'].unique())
        for matrix_size in matrix_sizes:
            size_data = df[df['Problem_Size'] == matrix_size]
            avg_speedup = size_data['Guided_Speedup'].mean()
            
            if matrix_size >= 1000000:
                size_label = f'{matrix_size/1000000:.0f}M elements'
            elif matrix_size >= 1000:
                size_label = f'{matrix_size/1000:.0f}K elements'
            else:
                size_label = f'{matrix_size:.0f} elements'
            
            print(f"  • {size_label:<12}: {avg_speedup:>6.1f}x average speedup")
        
        # Best performing configurations
        best_config = df.loc[df['Guided_Speedup'].idxmax()]
        print(f"\n🏆 Best Guided Performance Configuration:")
        size_label = f"{best_config['Problem_Size']/1000000:.0f}M" if best_config['Problem_Size'] >= 1000000 else f"{best_config['Problem_Size']/1000:.0f}K"
        print(f"  • Speedup: {best_config['Guided_Speedup']:.1f}x")
        print(f"  • Threads: {best_config['Threads']}")
        print(f"  • Matrix Size: {size_label} elements")
        print(f"  • Efficiency: {(best_config['Guided_Speedup'] / best_config['Threads']) * 100:.1f}%")
    
    print("\n� FIGURES 3A, 3B, 3C: THREAD-SPECIFIC ALGORITHM COMPARISON ANALYSIS")
    print("-" * 60)
    
    for thread_count in [8, 16, 96]:
        thread_data = df[df['Threads'] == thread_count].copy()
        if not thread_data.empty:
            print(f"\n📊 Analysis at {thread_count} Threads:")
            
            # Algorithm performance ranking
            algo_performance = {}
            for algo_col, algo_name in algorithms.items():
                if f'{algo_name}_Speedup' in thread_data.columns:
                    avg_speedup = thread_data[f'{algo_name}_Speedup'].mean()
                    max_speedup = thread_data[f'{algo_name}_Speedup'].max()
                    algo_performance[algo_name] = {'avg': avg_speedup, 'max': max_speedup}
            
            # Sort by average performance
            sorted_algos = sorted(algo_performance.items(), key=lambda x: x[1]['avg'], reverse=True)
            
            print(f"  🏆 Algorithm Ranking (by average speedup):")
            for i, (algo, perf) in enumerate(sorted_algos, 1):
                print(f"    {i}. {algo:<12}: {perf['avg']:>6.1f}x avg, {perf['max']:>6.1f}x max")
            
            # Problem size impact
            print(f"  📐 Problem Size Impact:")
            size_performance = thread_data.groupby('Problem_Size')['Speedup'].mean().sort_values(ascending=False)
            best_size = size_performance.index[0]
            worst_size = size_performance.index[-1]
            
            best_label = f"{best_size/1000000:.0f}M" if best_size >= 1000000 else f"{best_size/1000:.0f}K"
            worst_label = f"{worst_size/1000000:.0f}M" if worst_size >= 1000000 else f"{worst_size/1000:.0f}K"
            
            print(f"    • Best performing size: {best_label} elements ({size_performance.iloc[0]:.1f}x speedup)")
            print(f"    • Worst performing size: {worst_label} elements ({size_performance.iloc[-1]:.1f}x speedup)")
            print(f"    • Size impact factor: {size_performance.iloc[0] / size_performance.iloc[-1]:.1f}x improvement")
    
    print("\n🔍 FIGURE 4: PERFORMANCE HEATMAP ANALYSIS")
    print("-" * 60)
    
    # Overall performance trends
    print("📊 Overall Performance Patterns:")
    
    # Thread scaling analysis
    thread_performance = df.groupby('Threads')['Speedup'].agg(['mean', 'std', 'max']).round(1)
    print("\n🧵 Performance by Thread Count:")
    print(f"{'Threads':<8} {'Avg Speedup':<12} {'Std Dev':<10} {'Max Speedup':<12} {'Efficiency':<10}")
    print("-" * 60)
    for threads, row in thread_performance.iterrows():
        efficiency = (row['mean'] / threads) * 100
        print(f"{threads:<8} {row['mean']:<12} {row['std']:<10} {row['max']:<12} {efficiency:<10.1f}%")
    
    # Problem size scaling analysis
    print("\n📐 Performance by Problem Size Category:")
    size_performance = df.groupby('Problem_Size_Category')['Speedup'].agg(['mean', 'std', 'count']).round(1)
    print(f"{'Category':<12} {'Avg Speedup':<12} {'Std Dev':<10} {'Test Count':<10}")
    print("-" * 50)
    for category, row in size_performance.iterrows():
        print(f"{category:<12} {row['mean']:<12} {row['std']:<10} {row['count']:<10}")
    
    # Performance variability analysis
    print("\n📈 Key Performance Insights:")
    
    # Threading efficiency trend
    thread_efficiency = {threads: (df[df['Threads'] == threads]['Speedup'].mean() / threads) * 100 
                        for threads in sorted(df['Threads'].unique())}
    
    best_efficiency_threads = max(thread_efficiency, key=thread_efficiency.get)
    worst_efficiency_threads = min(thread_efficiency, key=thread_efficiency.get)
    
    print(f"  • Best threading efficiency: {best_efficiency_threads} threads ({thread_efficiency[best_efficiency_threads]:.1f}%)")
    print(f"  • Worst threading efficiency: {worst_efficiency_threads} threads ({thread_efficiency[worst_efficiency_threads]:.1f}%)")
    
    # Scalability analysis
    speedup_4 = df[df['Threads'] == 4]['Speedup'].mean()
    speedup_96 = df[df['Threads'] == 96]['Speedup'].mean()
    scaling_factor = speedup_96 / speedup_4
    
    print(f"  • Scaling factor (4→96 threads): {scaling_factor:.1f}x improvement")
    print(f"  • Overall performance range: {df['Speedup'].min():.1f}x to {df['Speedup'].max():.1f}x")
    
    # Algorithm consistency analysis
    algo_consistency = {}
    for algo_col, algo_name in algorithms.items():
        if f'{algo_name}_Speedup' in df.columns:
            std_dev = df[f'{algo_name}_Speedup'].std()
            mean_perf = df[f'{algo_name}_Speedup'].mean()
            cv = (std_dev / mean_perf) * 100  # Coefficient of variation
            algo_consistency[algo_name] = cv
    
    most_consistent = min(algo_consistency, key=algo_consistency.get)
    least_consistent = max(algo_consistency, key=algo_consistency.get)
    
    print(f"  • Most consistent algorithm: {most_consistent} (CV: {algo_consistency[most_consistent]:.1f}%)")
    print(f"  • Least consistent algorithm: {least_consistent} (CV: {algo_consistency[least_consistent]:.1f}%)")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    print("  1. Use Guided scheduling for optimal performance across all scenarios")
    print("  2. Target 16-32 threads for best efficiency/performance balance")
    print("  3. Larger problem sizes (>10M elements) show significantly better speedup")
    print("  4. Consider workload characteristics when selecting thread counts")
    print("  5. Monitor efficiency degradation beyond 32 threads for cost-effectiveness")
    
    print("\n" + "="*100)
    
    algorithms = {
        'Basic_Static_Time': 'Basic Static',
        'Dynamic_Time': 'Dynamic', 
        'Guided_Time': 'Guided',
        'Collapse_Time': 'Collapse'
    }
    
    # Calculate speedups for each algorithm
    for algo_col, algo_name in algorithms.items():
        if algo_col in df.columns:
            df[f'{algo_name}_Speedup'] = df['Sequential_Time'] / df[algo_col]
    
    print(f"\n📈 Dataset Overview:")
    print(f"  • Total data points: {len(df)}")
    print(f"  • Thread counts: {sorted(df['Threads'].unique())}")
    print(f"  • Problem size range: {df['Problem_Size'].min():,} - {df['Problem_Size'].max():,} elements")
    print(f"  • Sequential time range: {df['Sequential_Time'].min():.4f} - {df['Sequential_Time'].max():.4f} seconds")
    
    print(f"\n🏆 Best Overall Performance:")
    best_speedup_idx = df['Speedup'].idxmax()
    best_row = df.loc[best_speedup_idx]
    print(f"  • Best speedup: {best_row['Speedup']:.2f}x")
    print(f"  • Algorithm: {best_row['Best_Algorithm']}")
    print(f"  • Threads: {best_row['Threads']}")
    print(f"  • Problem size: {best_row['Problem_Size']:,} elements")
    
    print(f"\n📁 Generated Files:")
    print(f"  • 1_algorithm_selection_matrix.png - Best algorithm by problem size & threads")
    print(f"  • 2_speedup_vs_threads.png - Guided algorithm scaling by matrix size")
    print(f"  • 3a_algorithms_vs_problem_size_8threads.png - All algorithms vs problem size at 8 threads")
    print(f"  • 3b_algorithms_vs_problem_size_16threads.png - All algorithms vs problem size at 16 threads")
    print(f"  • 3c_algorithms_vs_problem_size_96threads.png - All algorithms vs problem size at 96 threads")
    print(f"  • 4_performance_heatmap.png - Average performance across all conditions")
    
    print(f"\n💡 Key Insights:")
    
    # Find best algorithm overall
    best_algo = None
    best_avg_speedup = 0
    for algo_col, algo_name in algorithms.items():
        if f'{algo_name}_Speedup' in df.columns:
            avg_speedup = df[f'{algo_name}_Speedup'].mean()
            if avg_speedup > best_avg_speedup:
                best_avg_speedup = avg_speedup
                best_algo = algo_name
    
    if best_algo:
        print(f"  • Best overall algorithm: {best_algo} (avg {best_avg_speedup:.2f}x speedup)")
    
    # Threading efficiency
    max_threads = df['Threads'].max()
    max_thread_speedup = df[df['Threads'] == max_threads]['Speedup'].mean()
    efficiency = (max_thread_speedup / max_threads) * 100
    print(f"  • Threading efficiency at {max_threads} threads: {efficiency:.1f}%")
    
    # Problem size impact
    large_problems = df[df['Problem_Size'] > df['Problem_Size'].median()]
    small_problems = df[df['Problem_Size'] <= df['Problem_Size'].median()]
    large_speedup = large_problems['Speedup'].mean()
    small_speedup = small_problems['Speedup'].mean()
    print(f"  • Large problems perform {large_speedup/small_speedup:.1f}x better than small problems")
    
    print("="*80)

def main():
    """Main function to generate all plots"""
    print("🎨 Creating High-Resolution Analysis Plots for HPC Report")
    print("="*60)
    
    csv_filename = 'performance_test.csv'
    output_dir = 'report_figures'
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = load_and_clean_data(csv_filename)
        print(f"✅ Loaded {len(df)} data points from {csv_filename}")
        print(f"📁 Output directory: {output_dir}")
        
        # Generate all plots
        print(f"\n🎯 Generating high-resolution plots...")
        plot1_algorithm_selection_matrix(df, output_dir)
        plot2_speedup_vs_threads(df, output_dir)
        plot3a_algorithms_vs_problem_size_8threads(df, output_dir)
        plot3b_algorithms_vs_problem_size_16threads(df, output_dir)
        plot3c_algorithms_vs_problem_size_96threads(df, output_dir)
        plot4_performance_heatmap(df, output_dir)
        
        # Create comprehensive analysis
        create_comprehensive_analysis(df, output_dir)
        
        print(f"\n🎉 All plots generated successfully!")
        print(f"📂 Find your high-resolution figures in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
