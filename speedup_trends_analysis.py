#!/usr/bin/env python3
"""
Focused Performance Analysis: Speedup Trends and Algorithm Selection
Shows average speedup for kernel size, matrix size, thread count, and best algorithm matrix
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

def create_speedup_trends_analysis(df, csv_filename):
    """Create focused speedup trends and algorithm selection analysis"""
    
    # Create figure with 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average Speedup vs Kernel Size (Top Left)
    kernel_stats = df.groupby('Kernel_H')['Speedup'].agg(['mean', 'std', 'count']).reset_index()
    
    ax1.errorbar(kernel_stats['Kernel_H'], kernel_stats['mean'], 
                yerr=kernel_stats['std'], fmt='o-', linewidth=3, markersize=8,
                capsize=5, capthick=2, color='blue', label='Average ± Std Dev')
    
    # Add trend line
    z = np.polyfit(kernel_stats['Kernel_H'], kernel_stats['mean'], 1)
    p = np.poly1d(z)
    ax1.plot(kernel_stats['Kernel_H'], p(kernel_stats['Kernel_H']), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
    
    ax1.set_xlabel('Kernel Size (N×N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Speedup', fontsize=12, fontweight='bold')
    ax1.set_title('Average Speedup vs Kernel Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value annotations
    for _, row in kernel_stats.iterrows():
        ax1.annotate(f'{row["mean"]:.1f}x\n(n={row["count"]})', 
                    (row['Kernel_H'], row['mean']), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontsize=10, fontweight='bold')
    
    # 2. Average Speedup vs Matrix Size (Top Right)
    matrix_stats = df.groupby('Matrix_H')['Speedup'].agg(['mean', 'std', 'count']).reset_index()
    
    ax2.errorbar(matrix_stats['Matrix_H'], matrix_stats['mean'], 
                yerr=matrix_stats['std'], fmt='s-', linewidth=3, markersize=8,
                capsize=5, capthick=2, color='green', label='Average ± Std Dev')
    
    # Add trend line
    z = np.polyfit(np.log(matrix_stats['Matrix_H']), matrix_stats['mean'], 1)
    trend_x = matrix_stats['Matrix_H']
    trend_y = z[0] * np.log(trend_x) + z[1]
    ax2.plot(trend_x, trend_y, "r--", alpha=0.8, linewidth=2, 
             label=f'Log Trend (slope: {z[0]:.2f})')
    
    ax2.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Speedup', fontsize=12, fontweight='bold')
    ax2.set_title('Average Speedup vs Matrix Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add value annotations for key points
    for i, row in matrix_stats.iterrows():
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            ax2.annotate(f'{row["mean"]:.1f}x', 
                        (row['Matrix_H'], row['mean']), 
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=9, fontweight='bold')
    
    # 3. Average Speedup vs Thread Count (Bottom Left)
    thread_stats = df.groupby('Threads')['Speedup'].agg(['mean', 'std', 'count']).reset_index()
    
    ax3.errorbar(thread_stats['Threads'], thread_stats['mean'], 
                yerr=thread_stats['std'], fmt='^-', linewidth=3, markersize=8,
                capsize=5, capthick=2, color='purple', label='Average ± Std Dev')
    
    # Add ideal speedup line
    max_threads = thread_stats['Threads'].max()
    ax3.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, linewidth=2, label='Ideal Speedup')
    
    # Add efficiency lines
    ax3.plot(thread_stats['Threads'], thread_stats['Threads'] * 0.5, 
             'orange', linestyle=':', alpha=0.7, linewidth=2, label='50% Efficiency')
    ax3.plot(thread_stats['Threads'], thread_stats['Threads'] * 0.75, 
             'red', linestyle=':', alpha=0.7, linewidth=2, label='75% Efficiency')
    
    ax3.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Speedup', fontsize=12, fontweight='bold')
    ax3.set_title('Average Speedup vs Thread Count', fontsize=14, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log', base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add value annotations
    for _, row in thread_stats.iterrows():
        efficiency = (row['mean'] / row['Threads']) * 100
        ax3.annotate(f'{row["mean"]:.1f}x\n({efficiency:.0f}%)', 
                    (row['Threads'], row['mean']), 
                    xytext=(0, 10), textcoords='offset points', 
                    ha='center', fontsize=9, fontweight='bold')
    
    # 4. Best Algorithm Matrix by Problem Size (Bottom Right)
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
        unique_algos = df['Best_Algorithm'].unique()
        algo_colors = {
            'Basicstatic': 0,
            'Dynamicscheduling': 1, 
            'Guidedscheduling': 2,
            'Collapseapproach': 3
        }
        
        # Convert to numerical values for plotting
        pivot_numeric = pivot_algo.applymap(lambda x: algo_colors.get(x, -1))
        
        # Create custom colormap
        colors = ['red', 'blue', 'green', 'orange']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors[:len(unique_algos)])
        
        im = ax4.imshow(pivot_numeric.values, cmap=cmap, aspect='auto', vmin=0, vmax=3)
        
        ax4.set_xticks(range(len(pivot_algo.columns)))
        ax4.set_xticklabels(pivot_algo.columns, fontsize=11)
        ax4.set_yticks(range(len(pivot_algo.index)))
        ax4.set_yticklabels(pivot_algo.index, fontsize=11)
        ax4.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Problem Size Category', fontsize=12, fontweight='bold')
        ax4.set_title('Best Algorithm by Problem Size & Thread Count', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot_algo.index)):
            for j in range(len(pivot_algo.columns)):
                if not pd.isna(pivot_numeric.iloc[i, j]):
                    algo_name = pivot_algo.iloc[i, j]
                    short_name = algo_name.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                    ax4.text(j, i, short_name, ha="center", va="center", 
                           color='white', fontweight='bold', fontsize=10)
        
        # Create legend
        legend_elements = []
        for algo, color_idx in algo_colors.items():
            if algo in unique_algos:
                clean_name = algo.replace('scheduling', '').replace('approach', '').replace('static', '').strip()
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[color_idx], label=clean_name))
        
        ax4.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('OpenMP 2D Convolution: Speedup Trends & Algorithm Selection', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    base = os.path.splitext(csv_filename)[0]
    os.makedirs(base, exist_ok=True)
    plt.savefig(os.path.join(base, 'speedup_trends_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def print_speedup_summary(df):
    """Print summary statistics for speedup trends"""
    print("\n" + "="*80)
    print("📊 SPEEDUP TRENDS SUMMARY")
    print("="*80)
    
    # Kernel size analysis
    kernel_stats = df.groupby('Kernel_H')['Speedup'].agg(['mean', 'std', 'min', 'max']).round(2)
    print("\n🔍 KERNEL SIZE ANALYSIS:")
    print(f"{'Kernel':<8} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 40)
    for kernel, stats in kernel_stats.iterrows():
        print(f"{kernel}x{kernel:<6} {stats['mean']:<8} {stats['std']:<8} {stats['min']:<8} {stats['max']:<8}")
    
    # Matrix size analysis  
    matrix_stats = df.groupby('Matrix_H')['Speedup'].agg(['mean', 'std', 'min', 'max']).round(2)
    print("\n📐 MATRIX SIZE ANALYSIS:")
    print(f"{'Matrix':<10} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 45)
    for matrix, stats in matrix_stats.iterrows():
        print(f"{matrix}x{matrix:<8} {stats['mean']:<8} {stats['std']:<8} {stats['min']:<8} {stats['max']:<8}")
    
    # Thread count analysis
    thread_stats = df.groupby('Threads').agg({
        'Speedup': ['mean', 'std', 'min', 'max'],
        'Threads': 'first'
    }).round(2)
    thread_stats.columns = ['mean', 'std', 'min', 'max', 'threads']
    thread_stats['efficiency'] = (thread_stats['mean'] / thread_stats['threads'] * 100).round(1)
    
    print("\n🧵 THREAD COUNT ANALYSIS:")
    print(f"{'Threads':<8} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Eff%':<8}")
    print("-" * 50)
    for threads, stats in thread_stats.iterrows():
        print(f"{threads:<8} {stats['mean']:<8} {stats['std']:<8} {stats['min']:<8} {stats['max']:<8} {stats['efficiency']:<8}")
    
    print("\n" + "="*80)

def main():
    """Main function"""
    print("📊 Speedup Trends & Algorithm Selection Analysis")
    print("="*60)
    
    csv_filename = 'performance_test.csv'
    try:
        df = load_and_clean_data(csv_filename)
        print(f"✅ Loaded {len(df)} data points from {csv_filename}")
        
        print("\n🎨 Creating speedup trends analysis...")
        create_speedup_trends_analysis(df, csv_filename)
        
        print_speedup_summary(df)
        
        base = os.path.splitext(csv_filename)[0]
        print(f"\n📁 Analysis saved to: {base}/speedup_trends_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
