#!/usr/bin/env python3
"""
按照ppl文件格式创建结构化的score JSON文件
"""

import json
import numpy as np
from datetime import datetime

def create_structured_json(input_file, output_file):
    """创建结构化的JSON文件"""
    # 读取score数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化新结构
    structured_data = {
        "summary": {
            "total_samples": data['statistics']['total_entries'],
            "rank_distribution": {}
        },
        "Low": {},
        "Medium": {},
        "High": {}
    }
    
    # 按rank分组数据
    rank_groups = {
        'Low': [],
        'Medium': [],
        'High': []
    }
    
    # 收集各组数据
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            rank = result_data.get('rank', 'Unknown')
            if rank in rank_groups:
                score = result_data['scores']['average']['score']
                abstraction = result_data.get('abstraction', '')
                
                rank_groups[rank].append({
                    'id': result_id,
                    'score': score,
                    'abstraction': abstraction
                })
    
    # 计算各组统计信息
    for rank, samples in rank_groups.items():
        if samples:
            scores = [s['score'] for s in samples]
            structured_data['summary']['rank_distribution'][rank] = {
                "count": len(samples),
                "mean_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores)
            }
            
            # 按question_id组织数据
            for sample in samples:
                structured_data[rank][sample['id']] = {
                    "abstraction": sample['abstraction'],
                    "average_score": sample['score']
                }
    
    # 添加创建信息
    structured_data['summary']['created_at'] = datetime.now().isoformat()
    structured_data['summary']['source_file'] = input_file
    structured_data['summary']['grouping_criteria'] = data['statistics']['rank_update_info']['criteria']
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    return structured_data

def print_summary(structured_data):
    """打印摘要信息"""
    print("=== 结构化JSON创建完成 ===")
    print(f"总样本数: {structured_data['summary']['total_samples']}")
    print()
    
    for rank in ['Low', 'Medium', 'High']:
        if rank in structured_data['summary']['rank_distribution']:
            stats = structured_data['summary']['rank_distribution'][rank]
            count = len(structured_data[rank])
            print(f"{rank}组:")
            print(f"  样本数: {stats['count']}")
            print(f"  平均分数: {stats['mean_score']:.3f}")
            print(f"  分数范围: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
            print(f"  实际存储: {count}个question_id")
            print()

def main():
    input_file = "traj_abs_format_score_rank.json"
    output_file = "traj_abs_format_score_structured.json"
    
    print("正在创建结构化JSON文件...")
    structured_data = create_structured_json(input_file, output_file)
    
    print_summary(structured_data)
    
    print(f"结构化JSON文件已保存为: {output_file}")
    print("\n文件结构:")
    print("- summary: 包含统计信息和分组标准")
    print("- Low: 低分组数据 (question_id -> abstraction + average_score)")
    print("- Medium: 中分组数据 (question_id -> abstraction + average_score)")
    print("- High: 高分组数据 (question_id -> abstraction + average_score)")

if __name__ == "__main__":
    main()
