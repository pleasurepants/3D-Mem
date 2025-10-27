#!/usr/bin/env python3
"""
寻找在保持区分度的同时尽可能均匀的三组划分方案
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

def find_balanced_cuts(scores):
    """寻找平衡的分割点"""
    n = len(scores)
    target_size = n // 3
    sorted_scores = np.sort(scores)
    
    print(f"总样本数: {n}")
    print(f"理想每组样本数: {target_size} (余数: {n % 3})")
    print()
    
    # 方案1: 严格按样本数均匀划分
    print("=== 方案1: 严格均匀划分 ===")
    idx1 = n // 3
    idx2 = 2 * n // 3
    cut1 = sorted_scores[idx1]
    cut2 = sorted_scores[idx2]
    
    group1 = scores[scores < cut1]
    group2 = scores[(scores >= cut1) & (scores < cut2)]
    group3 = scores[scores >= cut2]
    
    print(f"分割点: {cut1:.3f}, {cut2:.3f}")
    print(f"低分组: < {cut1:.3f} (样本数: {len(group1)})")
    print(f"中分组: {cut1:.3f} - {cut2:.3f} (样本数: {len(group2)})")
    print(f"高分组: >= {cut2:.3f} (样本数: {len(group3)})")
    print(f"各组均值: {np.mean(group1):.3f}, {np.mean(group2):.3f}, {np.mean(group3):.3f}")
    print(f"组间差异: {np.mean(group2) - np.mean(group1):.3f}, {np.mean(group3) - np.mean(group2):.3f}")
    print()
    
    # 方案2: 微调分割点以获得更好的平衡
    print("=== 方案2: 微调平衡 ===")
    # 尝试在严格均匀的基础上微调
    best_balance = float('inf')
    best_cuts = None
    best_groups = None
    
    # 在严格均匀分割点附近搜索
    for offset1 in range(-5, 6):
        for offset2 in range(-5, 6):
            new_idx1 = max(0, min(n-1, idx1 + offset1))
            new_idx2 = max(new_idx1+1, min(n-1, idx2 + offset2))
            
            new_cut1 = sorted_scores[new_idx1]
            new_cut2 = sorted_scores[new_idx2]
            
            g1 = scores[scores < new_cut1]
            g2 = scores[(scores >= new_cut1) & (scores < new_cut2)]
            g3 = scores[scores >= new_cut2]
            
            # 计算平衡度（样本数方差）
            sizes = [len(g1), len(g2), len(g3)]
            balance = np.var(sizes)
            
            # 计算区分度（组间均值差异）
            if len(g1) > 0 and len(g2) > 0 and len(g3) > 0:
                separation = (np.mean(g2) - np.mean(g1)) + (np.mean(g3) - np.mean(g2))
                
                # 综合评分：平衡度权重0.7，区分度权重0.3
                score = 0.7 * balance + 0.3 * (2.0 - separation)  # 2.0是理想区分度
                
                if score < best_balance:
                    best_balance = score
                    best_cuts = (new_cut1, new_cut2)
                    best_groups = (g1, g2, g3)
    
    cut1_opt, cut2_opt = best_cuts
    g1_opt, g2_opt, g3_opt = best_groups
    
    print(f"优化分割点: {cut1_opt:.3f}, {cut2_opt:.3f}")
    print(f"低分组: < {cut1_opt:.3f} (样本数: {len(g1_opt)})")
    print(f"中分组: {cut1_opt:.3f} - {cut2_opt:.3f} (样本数: {len(g2_opt)})")
    print(f"高分组: >= {cut2_opt:.3f} (样本数: {len(g3_opt)})")
    print(f"各组均值: {np.mean(g1_opt):.3f}, {np.mean(g2_opt):.3f}, {np.mean(g3_opt):.3f}")
    print(f"组间差异: {np.mean(g2_opt) - np.mean(g1_opt):.3f}, {np.mean(g3_opt) - np.mean(g2_opt):.3f}")
    print(f"样本数方差: {np.var([len(g1_opt), len(g2_opt), len(g3_opt)]):.2f}")
    print()
    
    # 方案3: 基于数据分布的自然均匀划分
    print("=== 方案3: 自然均匀划分 ===")
    # 观察数据分布，寻找更均匀的自然分割点
    unique_scores = np.unique(scores)
    print("分数分布详情:")
    for score in unique_scores:
        count = np.sum(scores == score)
        print(f"  {score:.3f}: {count}个样本")
    
    # 基于观察，尝试不同的分割点组合
    natural_options = [
        (2.75, 3.0),   # 基于数据自然分布
        (2.75, 3.25),  # 稍微调整
        (2.5, 3.0),    # 更均匀的低分组
        (2.5, 3.25),   # 平衡方案
    ]
    
    best_natural = None
    best_natural_score = float('inf')
    
    for cut1, cut2 in natural_options:
        g1 = scores[scores < cut1]
        g2 = scores[(scores >= cut1) & (scores < cut2)]
        g3 = scores[scores >= cut2]
        
        if len(g1) > 0 and len(g2) > 0 and len(g3) > 0:
            sizes = [len(g1), len(g2), len(g3)]
            balance = np.var(sizes)
            separation = (np.mean(g2) - np.mean(g1)) + (np.mean(g3) - np.mean(g2))
            score = 0.7 * balance + 0.3 * (2.0 - separation)
            
            if score < best_natural_score:
                best_natural_score = score
                best_natural = (cut1, cut2, g1, g2, g3)
    
    if best_natural:
        cut1_nat, cut2_nat, g1_nat, g2_nat, g3_nat = best_natural
        print(f"自然分割点: {cut1_nat:.3f}, {cut2_nat:.3f}")
        print(f"低分组: < {cut1_nat:.3f} (样本数: {len(g1_nat)})")
        print(f"中分组: {cut1_nat:.3f} - {cut2_nat:.3f} (样本数: {len(g2_nat)})")
        print(f"高分组: >= {cut2_nat:.3f} (样本数: {len(g3_nat)})")
        print(f"各组均值: {np.mean(g1_nat):.3f}, {np.mean(g2_nat):.3f}, {np.mean(g3_nat):.3f}")
        print(f"组间差异: {np.mean(g2_nat) - np.mean(g1_nat):.3f}, {np.mean(g3_nat) - np.mean(g2_nat):.3f}")
        print(f"样本数方差: {np.var([len(g1_nat), len(g2_nat), len(g3_nat)]):.2f}")
    
    return {
        'strict': (cut1, cut2, group1, group2, group3),
        'optimized': (cut1_opt, cut2_opt, g1_opt, g2_opt, g3_opt),
        'natural': best_natural if best_natural else None
    }

def create_balanced_visualization(scores, groupings):
    """创建平衡分组可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始分布
    axes[0,0].hist(scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0,0].set_title('Original Distribution')
    axes[0,0].set_xlabel('Average Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # 方案1: 严格均匀
    cut1, cut2, g1, g2, g3 = groupings['strict']
    axes[0,1].hist(g1, bins=10, alpha=0.7, color='red', label=f'Low (n={len(g1)})', edgecolor='black')
    axes[0,1].hist(g2, bins=10, alpha=0.7, color='yellow', label=f'Mid (n={len(g2)})', edgecolor='black')
    axes[0,1].hist(g3, bins=10, alpha=0.7, color='green', label=f'High (n={len(g3)})', edgecolor='black')
    axes[0,1].axvline(cut1, color='red', linestyle='--', alpha=0.7)
    axes[0,1].axvline(cut2, color='green', linestyle='--', alpha=0.7)
    axes[0,1].set_title('Strict Uniform')
    axes[0,1].set_xlabel('Average Score')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 方案2: 优化平衡
    cut1_opt, cut2_opt, g1_opt, g2_opt, g3_opt = groupings['optimized']
    axes[1,0].hist(g1_opt, bins=10, alpha=0.7, color='red', label=f'Low (n={len(g1_opt)})', edgecolor='black')
    axes[1,0].hist(g2_opt, bins=10, alpha=0.7, color='yellow', label=f'Mid (n={len(g2_opt)})', edgecolor='black')
    axes[1,0].hist(g3_opt, bins=10, alpha=0.7, color='green', label=f'High (n={len(g3_opt)})', edgecolor='black')
    axes[1,0].axvline(cut1_opt, color='red', linestyle='--', alpha=0.7)
    axes[1,0].axvline(cut2_opt, color='green', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Optimized Balance')
    axes[1,0].set_xlabel('Average Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 方案3: 自然均匀
    if groupings['natural']:
        cut1_nat, cut2_nat, g1_nat, g2_nat, g3_nat = groupings['natural']
        axes[1,1].hist(g1_nat, bins=10, alpha=0.7, color='red', label=f'Low (n={len(g1_nat)})', edgecolor='black')
        axes[1,1].hist(g2_nat, bins=10, alpha=0.7, color='yellow', label=f'Mid (n={len(g2_nat)})', edgecolor='black')
        axes[1,1].hist(g3_nat, bins=10, alpha=0.7, color='green', label=f'High (n={len(g3_nat)})', edgecolor='black')
        axes[1,1].axvline(cut1_nat, color='red', linestyle='--', alpha=0.7)
        axes[1,1].axvline(cut2_nat, color='green', linestyle='--', alpha=0.7)
        axes[1,1].set_title('Natural Uniform')
        axes[1,1].set_xlabel('Average Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'No natural option found', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Natural Uniform')
    
    plt.tight_layout()
    plt.savefig('balanced_groups_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    scores = load_scores()
    groupings = find_balanced_cuts(scores)
    create_balanced_visualization(scores, groupings)
    
    print("=== 推荐方案 ===")
    print("基于平衡度和区分度的综合考虑，推荐使用优化平衡方案:")
    
    cut1_opt, cut2_opt, g1_opt, g2_opt, g3_opt = groupings['optimized']
    print(f"- 低分组: < {cut1_opt:.3f} ({len(g1_opt)}个样本)")
    print(f"- 中分组: {cut1_opt:.3f} - {cut2_opt:.3f} ({len(g2_opt)}个样本)")
    print(f"- 高分组: >= {cut2_opt:.3f} ({len(g3_opt)}个样本)")
    
    print(f"\n各组均值: {np.mean(g1_opt):.3f}, {np.mean(g2_opt):.3f}, {np.mean(g3_opt):.3f}")
    print(f"组间差异: {np.mean(g2_opt) - np.mean(g1_opt):.3f}, {np.mean(g3_opt) - np.mean(g2_opt):.3f}")
    print(f"样本数方差: {np.var([len(g1_opt), len(g2_opt), len(g3_opt)]):.2f}")
    
    print(f"\n可视化图表已保存为: balanced_groups_comparison.png")

if __name__ == "__main__":
    main()
