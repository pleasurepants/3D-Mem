#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证rank字段的正确性
"""

import json

def verify_rank_field():
    """验证rank字段是否正确添加"""
    
    # 读取带rank的文件
    with open('/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl_rank.json', 'r', encoding='utf-8') as f:
        ranked_data = json.load(f)
    
    print("🔍 验证rank字段...")
    
    # 统计每个rank的样本数量和分数范围
    rank_stats = {}
    
    for sample_id, sample_data in ranked_data['results'].items():
        rank = sample_data['rank']
        perplexity = sample_data['perplexity']
        
        if rank not in rank_stats:
            rank_stats[rank] = {
                'count': 0,
                'scores': []
            }
        
        rank_stats[rank]['count'] += 1
        rank_stats[rank]['scores'].append(perplexity)
    
    print("\n📊 Rank字段验证结果:")
    print("-" * 50)
    
    for rank in ['Low', 'Medium', 'High']:
        if rank in rank_stats:
            stats = rank_stats[rank]
            scores = stats['scores']
            print(f"\n{rank}组:")
            print(f"  样本数量: {stats['count']}")
            print(f"  分数范围: {min(scores):.3f} - {max(scores):.3f}")
            print(f"  平均分数: {sum(scores)/len(scores):.3f}")
        else:
            print(f"\n❌ {rank}组未找到")
    
    # 验证分数范围是否符合预期
    print(f"\n🎯 分数范围验证:")
    print("-" * 50)
    
    low_scores = rank_stats['Low']['scores']
    medium_scores = rank_stats['Medium']['scores']
    high_scores = rank_stats['High']['scores']
    
    low_max = max(low_scores)
    medium_min = min(medium_scores)
    medium_max = max(medium_scores)
    high_min = min(high_scores)
    
    print(f"Low组最高分: {low_max:.3f}")
    print(f"Medium组最低分: {medium_min:.3f}")
    print(f"Medium组最高分: {medium_max:.3f}")
    print(f"High组最低分: {high_min:.3f}")
    
    # 检查是否有重叠
    if low_max < medium_min and medium_max < high_min:
        print("✅ 各组分数范围无重叠，划分正确")
    else:
        print("❌ 各组分数范围有重叠，需要检查")
        if low_max >= medium_min:
            print(f"   Low和Medium组重叠: {low_max:.3f} >= {medium_min:.3f}")
        if medium_max >= high_min:
            print(f"   Medium和High组重叠: {medium_max:.3f} >= {high_min:.3f}")
    
    print(f"\n✅ 验证完成！")

if __name__ == "__main__":
    verify_rank_field()
