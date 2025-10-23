#!/usr/bin/env python3
"""
寻找真正均匀的三组划分，优先考虑样本数均匀性
"""

import json
import numpy as np

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

def find_truly_balanced_groups(scores):
    """寻找真正均匀的分组"""
    n = len(scores)
    sorted_scores = np.sort(scores)
    
    print(f"总样本数: {n}")
    print(f"理想每组样本数: {n//3} (余数: {n%3})")
    print()
    
    # 方案1: 完全按样本数均匀划分（忽略分数连续性）
    print("=== 方案1: 完全均匀划分 ===")
    idx1 = n // 3
    idx2 = 2 * n // 3
    
    # 处理余数
    if n % 3 == 1:
        idx2 += 1  # 高分组多1个
    elif n % 3 == 2:
        idx1 += 1  # 中分组多1个
        idx2 += 1  # 高分组多1个
    
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
    print(f"样本数: {len(group1)}, {len(group2)}, {len(group3)}")
    print()
    
    # 方案2: 微调以获得更好的均匀性
    print("=== 方案2: 微调均匀性 ===")
    best_balance = float('inf')
    best_cuts = None
    best_groups = None
    
    # 在完全均匀的基础上微调
    for offset1 in range(-3, 4):
        for offset2 in range(-3, 4):
            new_idx1 = max(0, min(n-1, idx1 + offset1))
            new_idx2 = max(new_idx1+1, min(n-1, idx2 + offset2))
            
            new_cut1 = sorted_scores[new_idx1]
            new_cut2 = sorted_scores[new_idx2]
            
            g1 = scores[scores < new_cut1]
            g2 = scores[(scores >= new_cut1) & (scores < new_cut2)]
            g3 = scores[scores >= new_cut2]
            
            # 计算均匀性（样本数方差）
            sizes = [len(g1), len(g2), len(g3)]
            balance = np.var(sizes)
            
            # 计算区分度
            if len(g1) > 0 and len(g2) > 0 and len(g3) > 0:
                separation = (np.mean(g2) - np.mean(g1)) + (np.mean(g3) - np.mean(g2))
                
                # 优先考虑均匀性，权重0.8，区分度权重0.2
                score = 0.8 * balance + 0.2 * (2.0 - separation)
                
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
    print(f"样本数: {len(g1_opt)}, {len(g2_opt)}, {len(g3_opt)}")
    print(f"样本数方差: {np.var([len(g1_opt), len(g2_opt), len(g3_opt)]):.2f}")
    print()
    
    # 方案3: 基于数据分布寻找最均匀的自然分割
    print("=== 方案3: 自然均匀分割 ===")
    # 观察数据分布，寻找最均匀的分割点
    unique_scores = np.unique(scores)
    print("分数分布:")
    for score in unique_scores:
        count = np.sum(scores == score)
        print(f"  {score:.3f}: {count}个样本")
    
    # 尝试不同的分割点组合，优先考虑均匀性
    natural_options = [
        (2.5, 3.0),    # 尝试更均匀的低分组
        (2.75, 3.0),   # 稍微调整
        (2.5, 3.25),   # 平衡方案
        (2.75, 3.25),  # 基于观察
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
            
            # 优先考虑均匀性
            score = 0.8 * balance + 0.2 * (2.0 - separation)
            
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
        print(f"样本数: {len(g1_nat)}, {len(g2_nat)}, {len(g3_nat)}")
        print(f"样本数方差: {np.var([len(g1_nat), len(g2_nat), len(g3_nat)]):.2f}")
    
    return {
        'strict': (cut1, cut2, group1, group2, group3),
        'optimized': (cut1_opt, cut2_opt, g1_opt, g2_opt, g3_opt),
        'natural': best_natural if best_natural else None
    }

def main():
    scores = load_scores()
    groupings = find_truly_balanced_groups(scores)
    
    print("=== 最终推荐 ===")
    print("基于均匀性优先的原则，推荐以下方案:")
    
    # 选择最均匀的方案
    options = []
    for name, grouping in groupings.items():
        if grouping:
            cut1, cut2, g1, g2, g3 = grouping
            sizes = [len(g1), len(g2), len(g3)]
            balance = np.var(sizes)
            separation = (np.mean(g2) - np.mean(g1)) + (np.mean(g3) - np.mean(g2))
            options.append((name, balance, separation, grouping))
    
    # 按均匀性排序
    options.sort(key=lambda x: x[1])
    best_name, best_balance, best_separation, best_grouping = options[0]
    
    cut1, cut2, g1, g2, g3 = best_grouping
    print(f"方案: {best_name}")
    print(f"- 低分组: < {cut1:.3f} ({len(g1)}个样本, 平均{np.mean(g1):.3f})")
    print(f"- 中分组: {cut1:.3f} - {cut2:.3f} ({len(g2)}个样本, 平均{np.mean(g2):.3f})")
    print(f"- 高分组: >= {cut2:.3f} ({len(g3)}个样本, 平均{np.mean(g3):.3f})")
    print(f"样本数方差: {best_balance:.2f}")
    print(f"组间差异: {best_separation:.3f}")

if __name__ == "__main__":
    main()
