#!/usr/bin/env python3
"""
最终确定最平衡的三组划分方案
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

def find_most_balanced_groups(scores):
    """寻找最平衡的分组方案"""
    n = len(scores)
    target = n // 3
    
    print(f"总样本数: {n}")
    print(f"理想每组样本数: {target} (余数: {n%3})")
    print()
    
    # 尝试多种分割点组合
    options = [
        # (cut1, cut2, description)
        (2.5, 3.0, "激进低分组"),
        (2.75, 3.0, "适中低分组"),
        (2.5, 3.25, "激进低分组+标准中分组"),
        (2.75, 3.25, "适中低分组+标准中分组"),
        (2.5, 3.5, "激进低分组+宽中分组"),
        (2.75, 3.5, "适中低分组+宽中分组"),
        (3.0, 3.25, "原始方案"),
        (3.0, 3.5, "标准低分组+宽中分组"),
    ]
    
    results = []
    
    for cut1, cut2, desc in options:
        g1 = scores[scores < cut1]
        g2 = scores[(scores >= cut1) & (scores < cut2)]
        g3 = scores[scores >= cut2]
        
        if len(g1) > 0 and len(g2) > 0 and len(g3) > 0:
            sizes = [len(g1), len(g2), len(g3)]
            balance = np.var(sizes)
            separation = (np.mean(g2) - np.mean(g1)) + (np.mean(g3) - np.mean(g2))
            
            # 计算综合评分
            # 均匀性权重0.6，区分度权重0.4
            score = 0.6 * balance + 0.4 * (2.0 - separation)
            
            results.append({
                'cuts': (cut1, cut2),
                'description': desc,
                'groups': (g1, g2, g3),
                'sizes': sizes,
                'balance': balance,
                'separation': separation,
                'score': score
            })
    
    # 按综合评分排序
    results.sort(key=lambda x: x['score'])
    
    print("=== 所有方案对比 ===")
    for i, result in enumerate(results):
        cut1, cut2 = result['cuts']
        g1, g2, g3 = result['groups']
        sizes = result['sizes']
        
        print(f"{i+1}. {result['description']} ({cut1}, {cut2})")
        print(f"   样本数: 低{sizes[0]}, 中{sizes[1]}, 高{sizes[2]}")
        print(f"   各组均值: {np.mean(g1):.3f}, {np.mean(g2):.3f}, {np.mean(g3):.3f}")
        print(f"   样本数方差: {result['balance']:.2f}")
        print(f"   组间差异: {result['separation']:.3f}")
        print(f"   综合评分: {result['score']:.3f}")
        print()
    
    return results

def main():
    scores = load_scores()
    results = find_most_balanced_groups(scores)
    
    # 选择最佳方案
    best = results[0]
    cut1, cut2 = best['cuts']
    g1, g2, g3 = best['groups']
    
    print("=== 最终推荐方案 ===")
    print(f"方案: {best['description']}")
    print(f"分割点: {cut1}, {cut2}")
    print(f"- 低分组: < {cut1} ({len(g1)}个样本, 平均{np.mean(g1):.3f})")
    print(f"- 中分组: {cut1} - {cut2} ({len(g2)}个样本, 平均{np.mean(g2):.3f})")
    print(f"- 高分组: >= {cut2} ({len(g3)}个样本, 平均{np.mean(g3):.3f})")
    print(f"样本数方差: {best['balance']:.2f}")
    print(f"组间差异: {best['separation']:.3f}")
    print(f"综合评分: {best['score']:.3f}")
    
    # 如果用户想要更均匀的方案，提供备选
    print("\n=== 备选方案 ===")
    print("如果您更注重均匀性，可以考虑以下方案:")
    
    # 按均匀性排序
    uniform_results = sorted(results, key=lambda x: x['balance'])
    for i, result in enumerate(uniform_results[:3]):
        cut1, cut2 = result['cuts']
        g1, g2, g3 = result['groups']
        sizes = result['sizes']
        
        print(f"{i+1}. {result['description']} ({cut1}, {cut2})")
        print(f"   样本数: 低{sizes[0]}, 中{sizes[1]}, 高{sizes[2]}")
        print(f"   样本数方差: {result['balance']:.2f}")

if __name__ == "__main__":
    main()
