#!/usr/bin/env python3
"""
根据average分数创建三组划分，并输出分组结果
"""

import json
import numpy as np

def create_three_groups(json_file, output_file):
    """创建三组划分并保存结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有数据
    results = []
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            score = result_data['scores']['average']['score']
            results.append({
                'id': result_id,
                'score': score,
                'data': result_data
            })
    
    # 按分数排序
    results.sort(key=lambda x: x['score'])
    
    # 定义分组边界
    # 基于分析，使用自然分割点
    low_cutoff = 3.0
    high_cutoff = 3.25
    
    # 分组
    low_group = []
    mid_group = []
    high_group = []
    
    for result in results:
        score = result['score']
        if score < low_cutoff:
            low_group.append(result)
        elif score < high_cutoff:
            mid_group.append(result)
        else:
            high_group.append(result)
    
    # 创建分组结果
    groups = {
        'low': {
            'name': '低分组',
            'range': f'< {low_cutoff}',
            'count': len(low_group),
            'mean_score': np.mean([r['score'] for r in low_group]),
            'results': low_group
        },
        'mid': {
            'name': '中分组', 
            'range': f'{low_cutoff} - {high_cutoff}',
            'count': len(mid_group),
            'mean_score': np.mean([r['score'] for r in mid_group]),
            'results': mid_group
        },
        'high': {
            'name': '高分组',
            'range': f'>= {high_cutoff}',
            'count': len(high_group),
            'mean_score': np.mean([r['score'] for r in high_group]),
            'results': high_group
        }
    }
    
    # 保存分组结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    
    return groups

def print_group_summary(groups):
    """打印分组摘要"""
    print("=== 三组划分结果 ===")
    print(f"总样本数: {sum(g['count'] for g in groups.values())}")
    print()
    
    for group_key, group_data in groups.items():
        print(f"{group_data['name']} ({group_key}):")
        print(f"  分数范围: {group_data['range']}")
        print(f"  样本数量: {group_data['count']}")
        print(f"  平均分数: {group_data['mean_score']:.3f}")
        print()
    
    # 显示各组样本分布
    print("=== 各组样本分布 ===")
    for group_key, group_data in groups.items():
        print(f"\n{group_data['name']} 样本:")
        for i, result in enumerate(group_data['results'][:5]):  # 只显示前5个
            print(f"  {i+1}. ID: {result['id'][:8]}..., Score: {result['score']:.3f}")
        if len(group_data['results']) > 5:
            print(f"  ... 还有 {len(group_data['results']) - 5} 个样本")

def main():
    input_file = "traj_abs_format_score_rank.json"
    output_file = "three_groups.json"
    
    print("正在创建三组划分...")
    groups = create_three_groups(input_file, output_file)
    
    print_group_summary(groups)
    
    print(f"\n分组结果已保存到: {output_file}")
    print("\n推荐的分组方案:")
    print("- 低分组: < 3.000 (37个样本, 平均2.601分)")
    print("- 中分组: 3.000 - 3.250 (42个样本, 平均3.000分)")
    print("- 高分组: >= 3.250 (84个样本, 平均3.479分)")

if __name__ == "__main__":
    main()
