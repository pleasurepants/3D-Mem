#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成带rank字段的JSON文件
基于split_results.json中的分组信息，为原始数据添加rank字段
"""

import json
import numpy as np

def generate_ranked_json():
    """
    生成带rank字段的新JSON文件
    """
    
    # 读取原始数据
    print("Loading original data...")
    with open('/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 读取划分结果
    print("Loading split results...")
    with open('/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/split_results.json', 'r', encoding='utf-8') as f:
        split_results = json.load(f)
    
    # 获取最佳策略的分组信息
    best_strategy = split_results['best_strategy']
    strategy_data = split_results['strategies'][best_strategy]
    
    # 创建样本ID到组别的映射
    sample_to_group = {}
    for group_name, group_info in strategy_data['groups'].items():
        for sample_id in group_info['sample_ids']:
            sample_to_group[sample_id] = group_name
    
    print(f"Using strategy: {strategy_data['name']}")
    print(f"Total samples mapped: {len(sample_to_group)}")
    
    # 创建新的数据结构
    new_data = {
        'statistics': original_data['statistics'].copy(),
        'results': {}
    }
    
    # 为每个样本添加rank字段
    print("Adding rank fields...")
    for sample_id, sample_data in original_data['results'].items():
        new_sample_data = sample_data.copy()
        
        # 添加rank字段
        if sample_id in sample_to_group:
            new_sample_data['rank'] = sample_to_group[sample_id]
        else:
            print(f"Warning: Sample {sample_id} not found in split results")
            new_sample_data['rank'] = 'Unknown'
        
        new_data['results'][sample_id] = new_sample_data
    
    # 添加分组统计信息
    new_data['group_statistics'] = {}
    for group_name, group_info in strategy_data['groups'].items():
        new_data['group_statistics'][group_name] = {
            'count': group_info['count'],
            'percentage': (group_info['count'] / len(original_data['results'])) * 100,
            'statistics': group_info['statistics']
        }
    
    # 添加划分策略信息
    new_data['split_info'] = {
        'strategy_used': best_strategy,
        'strategy_name': strategy_data['name'],
        'total_samples': len(original_data['results']),
        'groups': list(strategy_data['groups'].keys())
    }
    
    # 保存新文件
    output_path = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl_rank.json'
    print(f"Saving ranked data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    # 验证结果
    print("\nVerification:")
    rank_counts = {}
    for sample_id, sample_data in new_data['results'].items():
        rank = sample_data['rank']
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    print("Rank distribution:")
    for rank, count in rank_counts.items():
        percentage = (count / len(new_data['results'])) * 100
        print(f"  {rank}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✅ Successfully generated ranked JSON file!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Total samples: {len(new_data['results'])}")
    print(f"🏷️  Added rank field to all samples")
    
    return output_path

def verify_ranked_data(file_path):
    """
    验证生成的ranked数据文件
    """
    print(f"\n🔍 Verifying ranked data file...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查基本结构
    required_keys = ['statistics', 'results', 'group_statistics', 'split_info']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing key: {key}")
            return False
        else:
            print(f"✅ Found key: {key}")
    
    # 检查每个样本是否有rank字段
    missing_rank = 0
    rank_distribution = {}
    
    for sample_id, sample_data in data['results'].items():
        if 'rank' not in sample_data:
            missing_rank += 1
        else:
            rank = sample_data['rank']
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
    
    if missing_rank > 0:
        print(f"❌ {missing_rank} samples missing rank field")
        return False
    else:
        print(f"✅ All samples have rank field")
    
    # 显示rank分布
    print(f"\n📊 Rank distribution:")
    for rank, count in rank_distribution.items():
        percentage = (count / len(data['results'])) * 100
        print(f"  {rank}: {count} samples ({percentage:.1f}%)")
    
    # 显示分组统计
    print(f"\n📈 Group statistics:")
    for group_name, group_stats in data['group_statistics'].items():
        print(f"  {group_name}: {group_stats['count']} samples ({group_stats['percentage']:.1f}%)")
        print(f"    Mean: {group_stats['statistics']['mean']:.3f}")
        print(f"    Range: {group_stats['statistics']['min']:.3f} - {group_stats['statistics']['max']:.3f}")
    
    print(f"\n✅ Verification completed successfully!")
    return True

if __name__ == "__main__":
    try:
        # 生成ranked JSON文件
        output_path = generate_ranked_json()
        
        # 验证生成的文件
        verify_ranked_data(output_path)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
