#!/usr/bin/env python3
"""
Algorithm Speedup Analysis: Individual algorithm performance vs problem size and threads
Shows speedup of each algorithm for different problem sizes and thread counts
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
    df = pd.read_csv(filename, comment='#')
    
    # Remove rows with missing or invalid data
    df = df.dropna(subset=['Sequential_Time', 'Best_Time', 'Speedup'])
    df = df[df['Sequential_Time'] > 0]
    df = df[df['Best_Time'] > 0]
    df = df[df['Speedup'] > 0]
    
    # Calculate problem size
    df['Problem_Size'] = df['Matrix_H'] * df['Matrix_W']
    
    return df

def create_algorithm_speedup_analysis(df, csv_filename):
    """Create algorithm speedup analysis with heatmap"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    
    # Define the grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, height_ratios=[1, 1, 1])
    
    # Calculate speedups for each algorithm
    algorithms = {
        'Basic_Static_Time': 'Basic Static',
        'Dynamic_Time': 'Dynamic', 
        'Guided_Time': 'Guided',
        'Collapse_Time': 'Collapse'
    }
    
    colors = ['red', 'blue', 'green', 'orange']
    
    # Calculate speedups for each algorithm
    for algo_col, algo_name in algorithms.items():
        if algo_col in df.columns:
            df[f'{algo_name}_Speedup'] = df['Sequential_Time'] / df[algo_col]
    
    # 1. Algorithm Speedup vs Problem Size (Top Row)
    for i, (algo_col, algo_name) in enumerate(algorithms.items()):
        ax = fig.add_subplot(gs[0, i])
        
        if f'{algo_name}_Speedup' in df.columns:
            # Group by problem size
            size_stats = df.groupby('Problem_Size')[f'{algo_name}_Speedup'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(size_stats['Problem_Size'], size_stats['mean'], 
                       yerr=size_stats['std'], fmt='o-', linewidth=2, markersize=6,
                       capsize=3, color=colors[i], label=f'{algo_name} Speedup')
            
            ax.set_xlabel('Problem Size (Matrix Elements)')
            ax.set_ylabel('Average Speedup')
            ax.set_title(f'{algo_name} Speedup vs Problem Size', fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add peak performance annotation
            max_idx = size_stats['mean'].idxmax()
            max_speedup = size_stats.loc[max_idx, 'mean']
            max_size = size_stats.loc[max_idx, 'Problem_Size']
            ax.annotate(f'Peak: {max_speedup:.1f}x', 
                       (max_size, max_speedup),
                       xytext=(10, 10), textcoords='offset points',
                       fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Algorithm Speedup vs Thread Count (Middle Row)
    for i, (algo_col, algo_name) in enumerate(algorithms.items()):
        ax = fig.add_subplot(gs[1, i])
        
        if f'{algo_name}_Speedup' in df.columns:
            # Group by thread count
            thread_stats = df.groupby('Threads')[f'{algo_name}_Speedup'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(thread_stats['Threads'], thread_stats['mean'], 
                       yerr=thread_stats['std'], fmt='s-', linewidth=2, markersize=6,
                       capsize=3, color=colors[i], label=f'{algo_name} Speedup')
            
            # Add ideal speedup line
            max_threads = thread_stats['Threads'].max()
            ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, linewidth=1, label='Ideal')
            
            # Add efficiency lines
            ax.plot(thread_stats['Threads'], thread_stats['Threads'] * 0.5, 
                   ':', alpha=0.5, color='gray', linewidth=1)
            ax.plot(thread_stats['Threads'], thread_stats['Threads'] * 0.75, 
                   ':', alpha=0.5, color='gray', linewidth=1)
            
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Average Speedup')
            ax.set_title(f'{algo_name} Speedup vs Thread Count', fontweight='bold')
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add efficiency annotations for key thread counts
            for _, row in thread_stats.iterrows():
                if row['Threads'] in [16, 32, 48, 96]:
                    efficiency = (row['mean'] / row['Threads']) * 100
                    ax.annotate(f'{efficiency:.0f}%', 
                               (row['Threads'], row['mean']),
                               xytext=(0, 5), textcoords='offset points',
                               ha='center', fontsize=8, fontweight='bold')
    
    # 3. Best Algorithm Heatmap (Bottom Row - spans all columns)
    ax_heatmap = fig.add_subplot(gs[2, :])
    
    # Create problem size categories for better visualization
    df['Problem_Size_Category'] = pd.cut(df['Problem_Size'], 
                                        bins=[0, 100000, 1000000, 10000000, 100000000, float('inf')],
                                        labels=['Small\n(<100K)', 'Medium\n(100K-1M)', 'Large\n(1M-10M)', 
                                               'Very Large\n(10M-100M)', 'Huge\n(>100M)'])
    
    # Create pivot table for best algorithm by problem size and thread count
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
        pivot_numeric = pivot_algo.applymap(lambda x: algo_colors_map.get(x, -1))
        
        # Create custom colormap
        colors_heat = ['red', 'blue', 'green', 'orange']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors_heat[:len(algo_colors_map)])
        
        im = ax_heatmap.imshow(pivot_numeric.values, cmap=cmap, aspect='auto', vmin=0, vmax=3)
        
        ax_heatmap.set_xticks(range(len(pivot_algo.columns)))
        ax_heatmap.set_xticklabels(pivot_algo.columns, fontsize=12)
        ax_heatmap.set_yticks(range(len(pivot_algo.index)))
        ax_heatmap.set_yticklabels(pivot_algo.index, fontsize=12)
        ax_heatmap.set_xlabel('Number of Threads', fontsize=14, fontweight='bold')
        ax_heatmap.set_ylabel('Problem Size Category', fontsize=14, fontweight='bold')
        ax_heatmap.set_title('Best Algorithm by Problem Size & Thread Count', fontsize=16, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_algo.index)):
            for j in range(len(pivot_algo.columns)):
                if not pd.isna(pivot_numeric.iloc[i, j]):
                    algo_name = pivot_algo.iloc[i, j]
                    short_name = algo_name.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                    ax_heatmap.text(j, i, short_name, ha="center", va="center", 
                                   color='white', fontweight='bold', fontsize=12)
        
        # Create legend
        legend_elements = []
        for algo, color_idx in algo_colors_map.items():
            if any(algo in unique_algo for unique_algo in df['Best_Algorithm'].unique()):
                clean_name = algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors_heat[color_idx], label=clean_name))
        
        ax_heatmap.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    # Add overall title
    fig.suptitle('OpenMP Algorithm Performance Analysis: Individual Speedups & Best Algorithm Selection', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the figure
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'algorithm_speedup_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_algorithm_comparison_table(df):
    """Create a detailed comparison table of algorithm performance"""
    print("\n" + "="*100)
    print("📊 DETAILED ALGORITHM PERFORMANCE COMPARISON")
    print("="*100)
    
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
    
    print("\n🔍 ALGORITHM PERFORMANCE BY THREAD COUNT:")
    print(f"{'Threads':<8} {'Basic Static':<12} {'Dynamic':<12} {'Guided':<12} {'Collapse':<12}")
    print("-" * 60)
    
    for threads in sorted(df['Threads'].unique()):
        thread_data = df[df['Threads'] == threads]
        row = f"{threads:<8}"
        
        for algo_col, algo_name in algorithms.items():
            if f'{algo_name}_Speedup' in thread_data.columns:
                avg_speedup = thread_data[f'{algo_name}_Speedup'].mean()
                row += f" {avg_speedup:>10.2f}x"
            else:
                row += f" {'N/A':>11}"
        print(row)
    
    print("\n📐 ALGORITHM PERFORMANCE BY PROBLEM SIZE:")
    print(f"{'Problem Size':<15} {'Basic Static':<12} {'Dynamic':<12} {'Guided':<12} {'Collapse':<12}")
    print("-" * 75)
    
    # Group by problem size ranges for readability
    size_ranges = [
        (0, 100000, "Small"),
        (100000, 1000000, "Medium"), 
        (1000000, 10000000, "Large"),
        (10000000, 100000000, "Very Large"),
        (100000000, float('inf'), "Huge")
    ]
    
    for min_size, max_size, size_label in size_ranges:
        size_data = df[(df['Problem_Size'] > min_size) & (df['Problem_Size'] <= max_size)]
        if not size_data.empty:
            row = f"{size_label:<15}"
            
            for algo_col, algo_name in algorithms.items():
                if f'{algo_name}_Speedup' in size_data.columns:
                    avg_speedup = size_data[f'{algo_name}_Speedup'].mean()
                    row += f" {avg_speedup:>10.2f}x"
                else:
                    row += f" {'N/A':>11}"
            print(row)
    
    print("\n🏆 BEST ALGORITHM RECOMMENDATIONS:")
    
    # Find best algorithm for each thread count
    print("\nBy Thread Count:")
    for threads in sorted(df['Threads'].unique()):
        thread_data = df[df['Threads'] == threads]
        best_speedup = 0
        best_algo = "None"
        
        for algo_col, algo_name in algorithms.items():
            if f'{algo_name}_Speedup' in thread_data.columns:
                avg_speedup = thread_data[f'{algo_name}_Speedup'].mean()
                if avg_speedup > best_speedup:
                    best_speedup = avg_speedup
                    best_algo = algo_name
        
        print(f"  {threads:>2} threads: {best_algo:<12} ({best_speedup:.2f}x average speedup)")
    
    print("\n" + "="*100)

def main():
    """Main function"""
    print("🔬 Algorithm Speedup Analysis")
    print("="*50)
    
    csv_filename = 'performance_test.csv'
    try:
        df = load_and_clean_data(csv_filename)
        print(f"✅ Loaded {len(df)} data points from {csv_filename}")
        
        print("\n🎨 Creating algorithm speedup analysis...")
        create_algorithm_speedup_analysis(df, csv_filename)
        
        create_algorithm_comparison_table(df)
        
        base = os.path.splitext(csv_filename)[0]
        print(f"\n📁 Analysis saved to: {base}/algorithm_speedup_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
