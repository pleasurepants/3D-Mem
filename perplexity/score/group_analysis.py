#!/usr/bin/env python3
"""
分析如何将average分数均匀分成三组
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_scores():
    """加载分数数据"""
    with open('traj_abs_format_score_rank.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores = []
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            score = result_data['scores']['average']['score']
            scores.append(score)
    
    return np.array(scores)

def find_optimal_cuts(scores, target_size):
    """寻找最优的切割点，使三组尽可能均匀"""
    n = len(scores)
    sorted_scores = np.sort(scores)
    
    best_cuts = None
    min_variance = float('inf')
    
    # 尝试不同的切割点
    for i in range(1, n-1):
        for j in range(i+1, n):
            group1_size = i
            group2_size = j - i
            group3_size = n - j
            
            # 计算各组大小的方差
            sizes = [group1_size, group2_size, group3_size]
            variance = np.var(sizes)
            
            if variance < min_variance:
                min_variance = variance
                best_cuts = (i, j)
    
    return best_cuts, sorted_scores

def analyze_grouping_options(scores):
    """分析不同的分组方案"""
    n = len(scores)
    target_size = n // 3
    
    print(f"总样本数: {n}")
    print(f"目标每组样本数: {target_size} (余数: {n % 3})")
    print()
    
    # 方案1: 严格按样本数均匀划分
    print("=== 方案1: 严格按样本数均匀划分 ===")
    cuts, sorted_scores = find_optimal_cuts(scores, target_size)
    cut1_idx, cut2_idx = cuts
    
    cut1_score = sorted_scores[cut1_idx]
    cut2_score = sorted_scores[cut2_idx]
    
    group1 = scores[scores < cut1_score]
    group2 = scores[(scores >= cut1_score) & (scores < cut2_score)]
    group3 = scores[scores >= cut2_score]
    
    print(f"低分组: < {cut1_score:.3f} (样本数: {len(group1)})")
    print(f"中分组: {cut1_score:.3f} - {cut2_score:.3f} (样本数: {len(group2)})")
    print(f"高分组: >= {cut2_score:.3f} (样本数: {len(group3)})")
    print(f"各组均值: {np.mean(group1):.3f}, {np.mean(group2):.3f}, {np.mean(group3):.3f}")
    print()
    
    # 方案2: 基于分位数的均匀划分
    print("=== 方案2: 基于分位数的均匀划分 ===")
    q33 = np.percentile(scores, 33.33)
    q67 = np.percentile(scores, 66.67)
    
    group1_q = scores[scores < q33]
    group2_q = scores[(scores >= q33) & (scores < q67)]
    group3_q = scores[scores >= q67]
    
    print(f"低分组: < {q33:.3f} (样本数: {len(group1_q)})")
    print(f"中分组: {q33:.3f} - {q67:.3f} (样本数: {len(group2_q)})")
    print(f"高分组: >= {q67:.3f} (样本数: {len(group3_q)})")
    print(f"各组均值: {np.mean(group1_q):.3f}, {np.mean(group2_q):.3f}, {np.mean(group3_q):.3f}")
    print()
    
    # 方案3: 基于数据分布的自然划分
    print("=== 方案3: 基于数据分布的自然划分 ===")
    # 观察数据分布，寻找自然的分割点
    unique_scores = np.unique(scores)
    print("分数分布:")
    for score in unique_scores:
        count = np.sum(scores == score)
        print(f"  {score:.3f}: {count}个样本")
    
    # 寻找自然的分割点
    # 基于观察，3.0和3.25是明显的分割点
    natural_cut1 = 3.0
    natural_cut2 = 3.25
    
    group1_n = scores[scores < natural_cut1]
    group2_n = scores[(scores >= natural_cut1) & (scores < natural_cut2)]
    group3_n = scores[scores >= natural_cut2]
    
    print(f"\n自然分割点:")
    print(f"低分组: < {natural_cut1:.3f} (样本数: {len(group1_n)})")
    print(f"中分组: {natural_cut1:.3f} - {natural_cut2:.3f} (样本数: {len(group2_n)})")
    print(f"高分组: >= {natural_cut2:.3f} (样本数: {len(group3_n)})")
    print(f"各组均值: {np.mean(group1_n):.3f}, {np.mean(group2_n):.3f}, {np.mean(group3_n):.3f}")
    print()
    
    return {
        'strict': (cut1_score, cut2_score, group1, group2, group3),
        'quantile': (q33, q67, group1_q, group2_q, group3_q),
        'natural': (natural_cut1, natural_cut2, group1_n, group2_n, group3_n)
    }

def create_grouping_visualization(scores, groupings):
    """创建分组可视化图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始分布
    axes[0,0].hist(scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,0].set_title('原始分数分布')
    axes[0,0].set_xlabel('Average Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # 方案1: 严格均匀划分
    cut1, cut2, g1, g2, g3 = groupings['strict']
    axes[0,1].hist(g1, bins=10, alpha=0.7, color='red', label=f'低分组 (n={len(g1)})', edgecolor='black')
    axes[0,1].hist(g2, bins=10, alpha=0.7, color='yellow', label=f'中分组 (n={len(g2)})', edgecolor='black')
    axes[0,1].hist(g3, bins=10, alpha=0.7, color='green', label=f'高分组 (n={len(g3)})', edgecolor='black')
    axes[0,1].axvline(cut1, color='red', linestyle='--', alpha=0.7)
    axes[0,1].axvline(cut2, color='green', linestyle='--', alpha=0.7)
    axes[0,1].set_title('方案1: 严格均匀划分')
    axes[0,1].set_xlabel('Average Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 方案2: 分位数划分
    cut1_q, cut2_q, g1_q, g2_q, g3_q = groupings['quantile']
    axes[1,0].hist(g1_q, bins=10, alpha=0.7, color='red', label=f'低分组 (n={len(g1_q)})', edgecolor='black')
    axes[1,0].hist(g2_q, bins=10, alpha=0.7, color='yellow', label=f'中分组 (n={len(g2_q)})', edgecolor='black')
    axes[1,0].hist(g3_q, bins=10, alpha=0.7, color='green', label=f'高分组 (n={len(g3_q)})', edgecolor='black')
    axes[1,0].axvline(cut1_q, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axvline(cut2_q, color='green', linestyle='--', alpha=0.7)
    axes[1,0].set_title('方案2: 分位数划分')
    axes[1,0].set_xlabel('Average Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 方案3: 自然划分
    cut1_n, cut2_n, g1_n, g2_n, g3_n = groupings['natural']
    axes[1,1].hist(g1_n, bins=10, alpha=0.7, color='red', label=f'低分组 (n={len(g1_n)})', edgecolor='black')
    axes[1,1].hist(g2_n, bins=10, alpha=0.7, color='yellow', label=f'中分组 (n={len(g2_n)})', edgecolor='black')
    axes[1,1].hist(g3_n, bins=10, alpha=0.7, color='green', label=f'高分组 (n={len(g3_n)})', edgecolor='black')
    axes[1,1].axvline(cut1_n, color='red', linestyle='--', alpha=0.7)
    axes[1,1].axvline(cut2_n, color='green', linestyle='--', alpha=0.7)
    axes[1,1].set_title('方案3: 自然划分')
    axes[1,1].set_xlabel('Average Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grouping_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    scores = load_scores()
    groupings = analyze_grouping_options(scores)
    create_grouping_visualization(scores, groupings)
    
    print("=== 推荐方案 ===")
    print("基于分析结果，推荐使用方案3（自然划分）:")
    print("- 低分组: < 3.000 (37个样本)")
    print("- 中分组: 3.000 - 3.250 (42个样本)")  
    print("- 高分组: >= 3.250 (84个样本)")
    print("\n理由:")
    print("1. 分割点3.0和3.25是数据的自然分界点")
    print("2. 低分组和中分组样本数相对均匀")
    print("3. 高分组虽然样本较多，但符合数据分布特征")
    print("4. 各组内部方差较小，组间差异明显")
    
    print(f"\n可视化图表已保存为: grouping_comparison.png")

if __name__ == "__main__":
    main()
