#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡划分分析脚本
分析perplexity分数分布，提供合理的三分组划分策略
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_perplexity_data(json_path):
    """
    从JSON文件中加载perplexity数据
    
    Args:
        json_path (str): JSON文件路径
        
    Returns:
        tuple: (perplexity_scores, statistics, results_dict)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取perplexity分数和对应的ID
    results_dict = {}
    for result_id, result_data in data['results'].items():
        results_dict[result_id] = {
            'perplexity': result_data['perplexity'],
            'abstraction': result_data['abstraction']
        }
    
    perplexity_scores = [result['perplexity'] for result in results_dict.values()]
    
    return perplexity_scores, data['statistics'], results_dict

def analyze_distribution(perplexity_scores):
    """
    分析perplexity分数的分布特征
    
    Args:
        perplexity_scores (list): perplexity分数列表
        
    Returns:
        dict: 分布分析结果
    """
    scores = np.array(perplexity_scores)
    
    # 基本统计
    stats = {
        'count': len(scores),
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'range': np.max(scores) - np.min(scores)
    }
    
    # 分位数
    percentiles = [10, 25, 33.33, 50, 66.67, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(scores, p)
    
    # 偏度和峰度
    from scipy import stats as scipy_stats
    stats['skewness'] = scipy_stats.skew(scores)
    stats['kurtosis'] = scipy_stats.kurtosis(scores)
    
    return stats

def evaluate_split_strategies(perplexity_scores, results_dict):
    """
    评估不同的划分策略
    
    Args:
        perplexity_scores (list): perplexity分数列表
        results_dict (dict): 结果字典
        
    Returns:
        dict: 各种策略的评估结果
    """
    scores = np.array(perplexity_scores)
    n_samples = len(scores)
    
    strategies = {}
    
    # 策略1: 等量划分 (Equal Count)
    group_size = n_samples // 3
    remainder = n_samples % 3
    
    # 按分数排序
    sorted_indices = np.argsort(scores)
    
    # 等量划分
    group1_indices = sorted_indices[:group_size + (1 if remainder > 0 else 0)]
    group2_indices = sorted_indices[group_size + (1 if remainder > 0 else 0):2*group_size + (2 if remainder > 1 else 1 if remainder > 0 else 0)]
    group3_indices = sorted_indices[2*group_size + (2 if remainder > 1 else 1 if remainder > 0 else 0):]
    
    strategies['equal_count'] = {
        'name': 'Equal Count Split',
        'groups': {
            'Low': {'indices': group1_indices, 'scores': scores[group1_indices]},
            'Medium': {'indices': group2_indices, 'scores': scores[group2_indices]},
            'High': {'indices': group3_indices, 'scores': scores[group3_indices]}
        }
    }
    
    # 策略2: 等分位数划分 (Equal Percentile)
    p33 = np.percentile(scores, 33.33)
    p67 = np.percentile(scores, 66.67)
    
    low_mask = scores <= p33
    medium_mask = (scores > p33) & (scores <= p67)
    high_mask = scores > p67
    
    strategies['equal_percentile'] = {
        'name': 'Equal Percentile Split',
        'groups': {
            'Low': {'indices': np.where(low_mask)[0], 'scores': scores[low_mask]},
            'Medium': {'indices': np.where(medium_mask)[0], 'scores': scores[medium_mask]},
            'High': {'indices': np.where(high_mask)[0], 'scores': scores[high_mask]}
        }
    }
    
    # 策略3: 混合策略 - 平衡数量和分数范围
    # 目标：每组数量相近，但分数范围相对均匀
    target_size = n_samples // 3
    
    # 使用K-means聚类思想，但手动调整边界
    sorted_scores = np.sort(scores)
    
    # 找到两个分割点，使得每组数量尽可能接近
    best_split1 = target_size
    best_split2 = 2 * target_size
    
    # 微调分割点以平衡数量和分数分布
    best_balance = float('inf')
    for i in range(max(1, target_size - 5), min(n_samples - 1, target_size + 5)):
        for j in range(max(i + 1, 2 * target_size - 5), min(n_samples, 2 * target_size + 5)):
            group1_scores = sorted_scores[:i]
            group2_scores = sorted_scores[i:j]
            group3_scores = sorted_scores[j:]
            
            # 计算平衡度（数量方差 + 分数范围方差）
            count_variance = np.var([len(group1_scores), len(group2_scores), len(group3_scores)])
            range_variance = np.var([
                np.max(group1_scores) - np.min(group1_scores),
                np.max(group2_scores) - np.min(group2_scores),
                np.max(group3_scores) - np.min(group3_scores)
            ])
            
            balance_score = count_variance + range_variance * 0.1  # 权重调整
            
            if balance_score < best_balance:
                best_balance = balance_score
                best_split1 = i
                best_split2 = j
    
    # 应用最佳分割
    group1_indices = sorted_indices[:best_split1]
    group2_indices = sorted_indices[best_split1:best_split2]
    group3_indices = sorted_indices[best_split2:]
    
    strategies['balanced'] = {
        'name': 'Balanced Split (Count + Range)',
        'groups': {
            'Low': {'indices': group1_indices, 'scores': scores[group1_indices]},
            'Medium': {'indices': group2_indices, 'scores': scores[group2_indices]},
            'High': {'indices': group3_indices, 'scores': scores[group3_indices]}
        }
    }
    
    return strategies

def calculate_group_statistics(strategies):
    """
    计算每个策略下各组的统计信息
    
    Args:
        strategies (dict): 策略字典
        
    Returns:
        dict: 包含统计信息的策略字典
    """
    for strategy_name, strategy in strategies.items():
        for group_name, group_data in strategy['groups'].items():
            scores = group_data['scores']
            group_data['stats'] = {
                'count': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'range': np.max(scores) - np.min(scores),
                'median': np.median(scores)
            }
    
    return strategies

def plot_split_comparison(strategies, output_path=None):
    """
    绘制不同策略的对比图
    
    Args:
        strategies (dict): 策略字典
        output_path (str): 输出图片路径
    """
    n_strategies = len(strategies)
    fig, axes = plt.subplots(2, n_strategies, figsize=(5*n_strategies, 10))
    
    if n_strategies == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    group_names = ['Low', 'Medium', 'High']
    
    for i, (strategy_name, strategy) in enumerate(strategies.items()):
        # 散点图
        ax1 = axes[0, i]
        y_pos = 0
        for j, (group_name, group_data) in enumerate(strategy['groups'].items()):
            scores = group_data['scores']
            x_pos = np.arange(len(scores)) + y_pos
            ax1.scatter(x_pos, scores, c=colors[j], alpha=0.7, s=30, label=group_name)
            y_pos += len(scores)
        
        ax1.set_title(f'{strategy["name"]}\nScatter Plot')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Perplexity Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = axes[1, i]
        box_data = [group_data['scores'] for group_data in strategy['groups'].values()]
        bp = ax2.boxplot(box_data, labels=group_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title(f'{strategy["name"]}\nBox Plot')
        ax2.set_ylabel('Perplexity Score')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    plt.show()

def print_strategy_analysis(strategies):
    """
    打印策略分析结果
    
    Args:
        strategies (dict): 策略字典
    """
    print("\n" + "="*80)
    print("SPLIT STRATEGY ANALYSIS")
    print("="*80)
    
    for strategy_name, strategy in strategies.items():
        print(f"\n📊 {strategy['name']}")
        print("-" * 50)
        
        total_count = sum(len(group['scores']) for group in strategy['groups'].values())
        count_variance = np.var([len(group['scores']) for group in strategy['groups'].values()])
        range_variance = np.var([group['stats']['range'] for group in strategy['groups'].values()])
        
        print(f"Total samples: {total_count}")
        print(f"Count variance: {count_variance:.2f}")
        print(f"Range variance: {range_variance:.2f}")
        
        for group_name, group_data in strategy['groups'].items():
            stats = group_data['stats']
            print(f"\n  {group_name} Group:")
            print(f"    Count: {stats['count']} ({stats['count']/total_count*100:.1f}%)")
            print(f"    Mean: {stats['mean']:.3f}")
            print(f"    Std: {stats['std']:.3f}")
            print(f"    Range: {stats['range']:.3f}")
            print(f"    Min-Max: {stats['min']:.3f} - {stats['max']:.3f}")

def recommend_best_strategy(strategies):
    """
    推荐最佳策略
    
    Args:
        strategies (dict): 策略字典
        
    Returns:
        str: 推荐策略名称
    """
    scores = {}
    
    for strategy_name, strategy in strategies.items():
        # 计算平衡度分数
        counts = [len(group['scores']) for group in strategy['groups'].values()]
        ranges = [group['stats']['range'] for group in strategy['groups'].values()]
        
        # 数量平衡度（越小越好）
        count_balance = np.std(counts) / np.mean(counts)
        
        # 分数范围平衡度（越小越好）
        range_balance = np.std(ranges) / np.mean(ranges)
        
        # 组间区分度（越大越好）
        means = [group['stats']['mean'] for group in strategy['groups'].values()]
        separation = (means[2] - means[0]) / np.mean(means)
        
        # 综合分数（越小越好）
        total_score = count_balance + range_balance * 0.5 - separation * 0.3
        
        scores[strategy_name] = {
            'count_balance': count_balance,
            'range_balance': range_balance,
            'separation': separation,
            'total_score': total_score
        }
    
    # 选择总分最低的策略
    best_strategy = min(scores.keys(), key=lambda x: scores[x]['total_score'])
    
    print(f"\n🏆 RECOMMENDED STRATEGY: {strategies[best_strategy]['name']}")
    print("-" * 50)
    print("Reasoning:")
    print(f"  • Count balance: {scores[best_strategy]['count_balance']:.3f}")
    print(f"  • Range balance: {scores[best_strategy]['range_balance']:.3f}")
    print(f"  • Group separation: {scores[best_strategy]['separation']:.3f}")
    print(f"  • Total score: {scores[best_strategy]['total_score']:.3f}")
    
    return best_strategy

def export_split_results(strategies, best_strategy, results_dict, output_path):
    """
    导出划分结果到JSON文件
    
    Args:
        strategies (dict): 策略字典
        best_strategy (str): 最佳策略名称
        results_dict (dict): 原始结果字典
        output_path (str): 输出文件路径
    """
    result_ids = list(results_dict.keys())
    
    export_data = {
        'best_strategy': best_strategy,
        'strategies': {}
    }
    
    for strategy_name, strategy in strategies.items():
        export_data['strategies'][strategy_name] = {
            'name': strategy['name'],
            'groups': {}
        }
        
        for group_name, group_data in strategy['groups'].items():
            group_ids = [result_ids[i] for i in group_data['indices']]
            export_data['strategies'][strategy_name]['groups'][group_name] = {
                'sample_ids': group_ids,
                'count': len(group_ids),
                'statistics': group_data['stats']
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSplit results exported to: {output_path}")

def main():
    """主函数"""
    # 输入文件路径
    json_path = "/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/traj_abs_format_ppl.json"
    
    # 输出文件路径
    comparison_plot_path = "/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/split_strategies_comparison.png"
    results_path = "/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/traj_abs_format/sim/split_results.json"
    
    print("Loading data...")
    try:
        perplexity_scores, statistics, results_dict = load_perplexity_data(json_path)
        print(f"Successfully loaded {len(perplexity_scores)} perplexity scores")
        
        print("\nAnalyzing distribution...")
        dist_stats = analyze_distribution(perplexity_scores)
        print(f"Distribution analysis completed")
        
        print("\nEvaluating split strategies...")
        strategies = evaluate_split_strategies(perplexity_scores, results_dict)
        strategies = calculate_group_statistics(strategies)
        
        print("\nGenerating comparison plots...")
        plot_split_comparison(strategies, comparison_plot_path)
        
        print_strategy_analysis(strategies)
        best_strategy = recommend_best_strategy(strategies)
        
        print("\nExporting results...")
        export_split_results(strategies, best_strategy, results_dict, results_path)
        
        print("\n✅ Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
