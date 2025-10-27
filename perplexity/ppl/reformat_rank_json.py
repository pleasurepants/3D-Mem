#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新排版rank JSON文件
按照rank分组，每个rank下面包含对应的question_id
"""

import json
from collections import defaultdict

def reformat_rank_json():
    """
    重新排版JSON文件结构
    """
    
    # 读取带rank的文件
    print("Loading ranked data...")
    with open('/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl_rank.json', 'r', encoding='utf-8') as f:
        ranked_data = json.load(f)
    
    # 创建新的数据结构
    new_data = {
        "Low": {},
        "Medium": {},
        "High": {}
    }
    
    # 按rank分组
    print("Reorganizing data by rank...")
    for question_id, question_data in ranked_data['results'].items():
        rank = question_data['rank']
        
        # 只保留abstraction和perplexity字段
        new_data[rank][question_id] = {
            "abstraction": question_data['abstraction'],
            "perplexity": question_data['perplexity']
        }
    
    # 统计信息
    stats = {}
    for rank in ['Low', 'Medium', 'High']:
        count = len(new_data[rank])
        perplexities = [data['perplexity'] for data in new_data[rank].values()]
        
        stats[rank] = {
            "count": count,
            "mean_perplexity": sum(perplexities) / len(perplexities) if perplexities else 0,
            "min_perplexity": min(perplexities) if perplexities else 0,
            "max_perplexity": max(perplexities) if perplexities else 0
        }
    
    # 添加统计信息到文件开头
    final_data = {
        "summary": {
            "total_samples": sum(stats[rank]['count'] for rank in stats),
            "rank_distribution": stats
        },
        **new_data
    }
    
    # 保存新文件
    output_path = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl_rank_reformatted.json'
    print(f"Saving reformatted data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # 验证结果
    print("\nVerification:")
    print("=" * 50)
    for rank in ['Low', 'Medium', 'High']:
        count = len(final_data[rank])
        mean_ppl = stats[rank]['mean_perplexity']
        min_ppl = stats[rank]['min_perplexity']
        max_ppl = stats[rank]['max_perplexity']
        
        print(f"{rank}组:")
        print(f"  样本数量: {count}")
        print(f"  平均perplexity: {mean_ppl:.3f}")
        print(f"  分数范围: {min_ppl:.3f} - {max_ppl:.3f}")
        print()
    
    print(f"✅ 重新排版完成！")
    print(f"📁 输出文件: {output_path}")
    print(f"📊 总样本数: {final_data['summary']['total_samples']}")
    
    return output_path

def verify_reformatted_data(file_path):
    """
    验证重新排版后的数据
    """
    print(f"\n🔍 验证重新排版后的数据...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查结构
    required_keys = ['summary', 'Low', 'Medium', 'High']
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing key: {key}")
            return False
        else:
            print(f"✅ Found key: {key}")
    
    # 检查每个rank组
    total_samples = 0
    for rank in ['Low', 'Medium', 'High']:
        count = len(data[rank])
        total_samples += count
        print(f"✅ {rank}组: {count} 个样本")
        
        # 检查每个样本的结构
        if count > 0:
            sample_id = list(data[rank].keys())[0]
            sample_data = data[rank][sample_id]
            required_sample_keys = ['abstraction', 'perplexity']
            
            for key in required_sample_keys:
                if key not in sample_data:
                    print(f"❌ Sample missing key: {key}")
                    return False
    
    print(f"✅ 总样本数: {total_samples}")
    print(f"✅ 数据验证完成！")
    
    return True

if __name__ == "__main__":
    try:
        # 重新排版JSON文件
        output_path = reformat_rank_json()
        
        # 验证重新排版后的文件
        verify_reformatted_data(output_path)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
