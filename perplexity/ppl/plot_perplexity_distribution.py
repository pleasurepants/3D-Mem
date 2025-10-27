#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perplexity分数分布可视化脚本
分析traj_abs_format_ppl.json文件中的perplexity分数并绘制散点图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_perplexity_data(json_path):
    """
    从JSON文件中加载perplexity数据
    
    Args:
        json_path (str): JSON文件路径
        
    Returns:
        tuple: (perplexity_scores, statistics)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取perplexity分数
    perplexity_scores = []
    for result_id, result_data in data['results'].items():
        perplexity_scores.append(result_data['perplexity'])
    
    return perplexity_scores, data['statistics']

def plot_perplexity_distribution(perplexity_scores, statistics, output_path=None):
    """
    绘制perplexity分数分布散点图
    
    Args:
        perplexity_scores (list): perplexity分数列表
        statistics (dict): 统计信息
        output_path (str): 输出图片路径
    """
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(perplexity_scores)))
    
    # 第一个子图：散点图
    x_positions = range(len(perplexity_scores))
    scatter = ax1.scatter(x_positions, perplexity_scores, c=colors, alpha=0.7, s=50)
    
    # 添加统计线
    mean_val = statistics['mean']
    median_val = statistics['median']
    std_val = statistics['std']
    
    ax1.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax1.axhline(y=median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax1.axhline(y=mean_val + std_val, color='gray', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.2f}')
    ax1.axhline(y=mean_val - std_val, color='gray', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.2f}')
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Perplexity Score')
    ax1.set_title('Perplexity Score Distribution Scatter Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：直方图
    ax2.hist(perplexity_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax2.axvline(x=median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax2.set_xlabel('Perplexity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Perplexity Score Distribution Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print("\n=== Perplexity Score Statistics ===")
    print(f"Total samples: {statistics['total_entries']}")
    print(f"Mean: {statistics['mean']:.4f}")
    print(f"Median: {statistics['median']:.4f}")
    print(f"Standard deviation: {statistics['std']:.4f}")
    print(f"Minimum: {statistics['min']:.4f}")
    print(f"Maximum: {statistics['max']:.4f}")
    
    # 计算分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n=== Percentile Information ===")
    for p in percentiles:
        value = np.percentile(perplexity_scores, p)
        print(f"{p}th percentile: {value:.4f}")

def main():
    """主函数"""
    # 输入文件路径
    json_path = "/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl.json"
    
    # 输出图片路径
    output_path = "/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/perplexity_distribution.png"
    
    print("Loading data...")
    try:
        perplexity_scores, statistics = load_perplexity_data(json_path)
        print(f"Successfully loaded {len(perplexity_scores)} perplexity scores")
        
        print("Generating visualization...")
        plot_perplexity_distribution(perplexity_scores, statistics, output_path)
        
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file format")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
