#!/usr/bin/env python3
"""
根据最佳平衡方案更新rank字段
"""

import json
import shutil
from datetime import datetime

def update_ranks_balanced(input_file, output_file):
    """使用最佳平衡方案更新rank字段"""
    # 备份原文件
    backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(input_file, backup_file)
    print(f"原文件已备份为: {backup_file}")
    
    # 读取原文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 使用最佳平衡方案的分割点
    low_cutoff = 3.0
    high_cutoff = 3.5
    
    # 统计更新情况
    stats = {
        'low': 0,
        'medium': 0, 
        'high': 0,
        'unchanged': 0,
        'changed': 0
    }
    
    # 更新每个结果的rank
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            score = result_data['scores']['average']['score']
            old_rank = result_data.get('rank', 'Unknown')
            
            # 根据分数确定新rank
            if score < low_cutoff:
                new_rank = 'Low'
                stats['low'] += 1
            elif score < high_cutoff:
                new_rank = 'Medium'
                stats['medium'] += 1
            else:
                new_rank = 'High'
                stats['high'] += 1
            
            # 更新rank字段
            result_data['rank'] = new_rank
            
            # 统计变化
            if old_rank != new_rank:
                stats['changed'] += 1
            else:
                stats['unchanged'] += 1
    
    # 更新统计信息
    data['statistics']['rank_distribution'] = {
        'Low': stats['low'],
        'Medium': stats['medium'],
        'High': stats['high']
    }
    
    data['statistics']['rank_update_info'] = {
        'updated_at': datetime.now().isoformat(),
        'method': 'balanced_groups',
        'criteria': {
            'Low': f'< {low_cutoff}',
            'Medium': f'{low_cutoff} - {high_cutoff}',
            'High': f'>= {high_cutoff}'
        },
        'changes': {
            'total_changed': stats['changed'],
            'total_unchanged': stats['unchanged']
        },
        'balance_info': {
            'sample_variance': 416.89,
            'group_differences': 1.096,
            'composite_score': 250.495
        }
    }
    
    # 保存更新后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return stats

def print_update_summary(stats):
    """打印更新摘要"""
    print("\n=== 平衡分组更新摘要 ===")
    print(f"Low组: {stats['low']} 个样本")
    print(f"Medium组: {stats['medium']} 个样本")
    print(f"High组: {stats['high']} 个样本")
    print(f"总计: {sum([stats['low'], stats['medium'], stats['high']])} 个样本")
    print()
    print(f"发生变化的样本: {stats['changed']} 个")
    print(f"未发生变化的样本: {stats['unchanged']} 个")
    
    # 计算均匀性
    sizes = [stats['low'], stats['medium'], stats['high']]
    import numpy as np
    variance = np.var(sizes)
    print(f"样本数方差: {variance:.2f}")
    print(f"各组样本数: {sizes}")

def verify_ranks(file_path):
    """验证rank更新结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n=== 验证结果 ===")
    
    # 按rank分组统计
    rank_stats = {'Low': [], 'Medium': [], 'High': []}
    
    for result_id, result_data in data['results'].items():
        if 'scores' in result_data and 'average' in result_data['scores']:
            rank = result_data.get('rank', 'Unknown')
            score = result_data['scores']['average']['score']
            rank_stats[rank].append(score)
    
    for rank, scores in rank_stats.items():
        if scores:
            import numpy as np
            print(f"{rank}组: {len(scores)}个样本, 分数范围: {min(scores):.3f} - {max(scores):.3f}, 平均: {np.mean(scores):.3f}")
        else:
            print(f"{rank}组: 0个样本")

def main():
    input_file = "traj_abs_format_score_rank.json"
    output_file = "traj_abs_format_score_rank_balanced.json"
    
    print("正在使用最佳平衡方案更新rank字段...")
    stats = update_ranks_balanced(input_file, output_file)
    
    print_update_summary(stats)
    verify_ranks(output_file)
    
    print(f"\n更新后的文件已保存为: {output_file}")
    print("新的平衡分组标准:")
    print("- Low: < 3.0 (37个样本)")
    print("- Medium: 3.0 - 3.5 (83个样本)") 
    print("- High: >= 3.5 (43个样本)")
    print("\n这个方案在保持区分度的同时，显著改善了样本数的均匀性！")

if __name__ == "__main__":
    main()
