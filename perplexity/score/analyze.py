#!/usr/bin/env python3
"""
分析traj_abs_format_score_rank.json中average分数的分布情况
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_and_analyze_scores(json_file):
    """加载JSON文件并分析average分数分布"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有average分数
    average_scores = []
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            score = result_data['scores']['average']['score']
            average_scores.append(score)
    
    return average_scores

def create_distribution_plot(scores, output_file):
    """创建分数分布图"""
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 直方图
    ax1.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Average Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Average Score Distribution (Histogram)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图
    ax2.boxplot(scores, vert=True)
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Score Distribution (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 密度图
    ax3.hist(scores, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Average Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Average Score Distribution (Density)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布图
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax4.plot(sorted_scores, cumulative, marker='o', markersize=2, linewidth=2)
    ax4.set_xlabel('Average Score')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Function')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(scores):
    """打印统计信息"""
    scores_array = np.array(scores)
    
    print("=== Average Score 统计分析 ===")
    print(f"总样本数: {len(scores)}")
    print(f"均值: {np.mean(scores_array):.4f}")
    print(f"中位数: {np.median(scores_array):.4f}")
    print(f"标准差: {np.std(scores_array):.4f}")
    print(f"最小值: {np.min(scores_array):.4f}")
    print(f"最大值: {np.max(scores_array):.4f}")
    print(f"25%分位数: {np.percentile(scores_array, 25):.4f}")
    print(f"75%分位数: {np.percentile(scores_array, 75):.4f}")
    
    # 分数区间统计
    print("\n=== 分数区间分布 ===")
    ranges = [
        (0, 1, "0-1"),
        (1, 2, "1-2"), 
        (2, 3, "2-3"),
        (3, 4, "3-4"),
        (4, 5, "4-5")
    ]
    
    for min_val, max_val, label in ranges:
        count = sum(1 for score in scores if min_val <= score < max_val)
        percentage = count / len(scores) * 100
        print(f"{label}: {count} 个样本 ({percentage:.1f}%)")
    
    # 特殊值统计
    print(f"\n满分(5.0): {scores.count(5.0)} 个样本")
    print(f"高分(≥4.0): {sum(1 for s in scores if s >= 4.0)} 个样本")
    print(f"中等(3.0-4.0): {sum(1 for s in scores if 3.0 <= s < 4.0)} 个样本")
    print(f"低分(<3.0): {sum(1 for s in scores if s < 3.0)} 个样本")

def main():
    json_file = "traj_abs_format_score_rank.json"
    output_file = "score_distribution.png"
    
    print("正在加载数据...")
    scores = load_and_analyze_scores(json_file)
    
    print(f"成功加载 {len(scores)} 个average分数")
    
    # 打印统计信息
    print_statistics(scores)
    
    # 创建分布图
    print(f"\n正在生成分布图: {output_file}")
    create_distribution_plot(scores, output_file)
    print("分析完成！")

if __name__ == "__main__":
    main()
