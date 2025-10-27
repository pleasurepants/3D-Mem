#!/usr/bin/env python3
"""
Script to analyze retrieval qids from log files and extract corresponding perplexity scores.
This script processes log files to extract retrieval information and match it with perplexity data.
"""

import json
import re
import os
import statistics
import sys
from typing import Dict, List, Tuple, Any
from pathlib import Path


def load_perplexity_data(ppl_file_path: str) -> Dict[str, float]:
    """Load perplexity data from JSON file."""
    with open(ppl_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    perplexity_map = {}
    for qid, info in data['results'].items():
        perplexity_map[qid] = info['perplexity']
    
    return perplexity_map


def extract_all_retrievals(log_file_path: str) -> List[Dict[str, Any]]:
    """Extract all retrieval information from log file."""
    retrievals = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for TrajSim selected lines
        if "[TrajSim] selected" in line and "question" in line:
            # Look for the next few lines containing qid information
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j].strip()
                if "qid=" in next_line:
                    # Extract qid from line like "00:00:50 -   #1: qid=5a8b3936-43e0-4474-ac15-efaf488265a1 | ..."
                    qid_match = re.search(r'qid=([a-f0-9-]+)', next_line)
                    if qid_match:
                        qid = qid_match.group(1)
                        retrievals.append({
                            'line_number': j + 1,
                            'qid': qid
                        })
                elif next_line.startswith("00:") and not next_line.startswith("#") and "qid=" not in next_line:
                    # Stop when we hit the next timestamp line that's not a qid line
                    break
    
    return retrievals


def extract_question_info(log_file_path: str) -> Dict[str, Dict[str, Any]]:
    """Extract question information from log file."""
    question_data = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract question initialization
        if "Question id" in line and "initialization successful" in line:
            match = re.search(r'Question id ([a-f0-9-]+) initialization successful', line)
            if match:
                question_id = match.group(1)
                question_data[question_id] = {
                    'initialization_line': i + 1,
                    'total_steps': 0,
                    'status': 'fail',  # Default to fail, will be changed to success if "finish successfully" is found
                    'success_line': None
                }
    
    # Check for fail status and count steps in each question's section
    sorted_questions = sorted(question_data.items(), key=lambda x: x[1]['initialization_line'])
    
    for i, (qid, info) in enumerate(sorted_questions):
        start_line = info['initialization_line']
        end_line = sorted_questions[i + 1][1]['initialization_line'] if i + 1 < len(sorted_questions) else len(lines)
        
        max_step = 0
        # Check for success and count steps in this question's section
        for j in range(start_line - 1, end_line):
            if j < len(lines):
                line = lines[j].strip()
                line_lower = line.lower()
                
                # Check for success
                if 'finish successfully' in line_lower:
                    question_data[qid]['status'] = 'success'
                    question_data[qid]['success_line'] = j + 1
                
                # Count steps
                if "== step:" in line:
                    match = re.search(r'== step: (\d+)', line)
                    if match:
                        step_num = int(match.group(1))
                        max_step = max(max_step, step_num)
        
        # Update total steps for this question
        question_data[qid]['total_steps'] = max_step
    
    return question_data


def associate_retrievals_with_questions(retrievals: List[Dict], question_data: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Associate retrievals with questions based on line numbers."""
    question_retrievals = {}
    
    # Initialize empty lists for each question
    for qid in question_data:
        question_retrievals[qid] = []
    
    # Sort questions by initialization line
    sorted_questions = sorted(question_data.items(), key=lambda x: x[1]['initialization_line'])
    
    for i, (qid, info) in enumerate(sorted_questions):
        start_line = info['initialization_line']
        end_line = sorted_questions[i + 1][1]['initialization_line'] if i + 1 < len(sorted_questions) else float('inf')
        
        # Find retrievals that occur between this question and the next
        for retrieval in retrievals:
            if start_line <= retrieval['line_number'] < end_line:
                question_retrievals[qid].append(retrieval)
    
    return question_retrievals


def calculate_perplexity_statistics(retrievals: List[Dict], perplexity_map: Dict[str, float]) -> Dict[str, Any]:
    """Calculate perplexity statistics for a list of retrievals."""
    perplexity_scores = []
    valid_retrievals = []
    
    for retrieval in retrievals:
        qid = retrieval['qid']
        if qid in perplexity_map:
            perplexity_scores.append(perplexity_map[qid])
            valid_retrievals.append({
                'qid': qid,
                'perplexity': perplexity_map[qid]
            })
    
    if not perplexity_scores:
        return {
            'min_perplexity': None,
            'max_perplexity': None,
            'mean_perplexity': None,
            'median_perplexity': None,
            'std_perplexity': None,
            'valid_retrievals': valid_retrievals,
            'total_retrievals': len(retrievals),
            'valid_retrievals_count': 0
        }
    
    return {
        'min_perplexity': min(perplexity_scores),
        'max_perplexity': max(perplexity_scores),
        'mean_perplexity': statistics.mean(perplexity_scores),
        'median_perplexity': statistics.median(perplexity_scores),
        'std_perplexity': statistics.stdev(perplexity_scores) if len(perplexity_scores) > 1 else 0.0,
        'valid_retrievals': valid_retrievals,
        'total_retrievals': len(retrievals),
        'valid_retrievals_count': len(perplexity_scores)
    }


def analyze_log_file(log_file_path: str, ppl_file_path: str, output_file_path: str):
    """Main function to analyze log file and generate output JSON."""
    print(f"Loading perplexity data from: {ppl_file_path}")
    perplexity_map = load_perplexity_data(ppl_file_path)
    print(f"Loaded {len(perplexity_map)} perplexity entries")
    
    print(f"Extracting retrieval data from: {log_file_path}")
    all_retrievals = extract_all_retrievals(log_file_path)
    print(f"Found {len(all_retrievals)} total retrievals")
    
    print(f"Extracting question information from: {log_file_path}")
    question_data = extract_question_info(log_file_path)
    print(f"Found {len(question_data)} questions")
    
    print("Associating retrievals with questions...")
    question_retrievals = associate_retrievals_with_questions(all_retrievals, question_data)
    
    # Calculate statistics for each question
    results = {}
    success_perplexities = []
    fail_perplexities = []
    success_steps = []
    fail_steps = []
    
    for question_id, data in question_data.items():
        print(f"Processing question: {question_id}")
        retrievals = question_retrievals[question_id]
        stats = calculate_perplexity_statistics(retrievals, perplexity_map)
        
        results[question_id] = {
            'summary': {
                'total_steps': data['total_steps'],
                'status': data['status'],
                'min_perplexity': stats['min_perplexity'],
                'max_perplexity': stats['max_perplexity'],
                'mean_perplexity': stats['mean_perplexity'],
                'median_perplexity': stats['median_perplexity'],
                'std_perplexity': stats['std_perplexity'],
                'total_retrievals': stats['total_retrievals'],
                'valid_retrievals_count': stats['valid_retrievals_count']
            },
            'detailed_retrievals': stats['valid_retrievals']
        }
        
        # Collect data for global statistics
        if data['status'] == 'success':
            success_steps.append(data['total_steps'])
            if stats['mean_perplexity'] is not None:
                success_perplexities.append(stats['mean_perplexity'])
        else:  # fail
            fail_steps.append(data['total_steps'])
            if stats['mean_perplexity'] is not None:
                fail_perplexities.append(stats['mean_perplexity'])
    
    # Calculate global statistics
    global_stats = {
        'success': {
            'count': len(success_steps),
            'steps': {
                'min': min(success_steps) if success_steps else None,
                'max': max(success_steps) if success_steps else None,
                'mean': statistics.mean(success_steps) if success_steps else None,
                'median': statistics.median(success_steps) if success_steps else None,
                'std': statistics.stdev(success_steps) if len(success_steps) > 1 else 0.0
            },
            'perplexity': {
                'min': min(success_perplexities) if success_perplexities else None,
                'max': max(success_perplexities) if success_perplexities else None,
                'mean': statistics.mean(success_perplexities) if success_perplexities else None,
                'median': statistics.median(success_perplexities) if success_perplexities else None,
                'std': statistics.stdev(success_perplexities) if len(success_perplexities) > 1 else 0.0
            }
        },
        'fail': {
            'count': len(fail_steps),
            'steps': {
                'min': min(fail_steps) if fail_steps else None,
                'max': max(fail_steps) if fail_steps else None,
                'mean': statistics.mean(fail_steps) if fail_steps else None,
                'median': statistics.median(fail_steps) if fail_steps else None,
                'std': statistics.stdev(fail_steps) if len(fail_steps) > 1 else 0.0
            },
            'perplexity': {
                'min': min(fail_perplexities) if fail_perplexities else None,
                'max': max(fail_perplexities) if fail_perplexities else None,
                'mean': statistics.mean(fail_perplexities) if fail_perplexities else None,
                'median': statistics.median(fail_perplexities) if fail_perplexities else None,
                'std': statistics.stdev(fail_perplexities) if len(fail_perplexities) > 1 else 0.0
            }
        }
    }
    
    # Add global statistics to results
    results['_global_statistics'] = global_stats
    
    # Save results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file_path}")
    print(f"Processed {len(results)} questions")


def main():
    """Main entry point."""
    if len(sys.argv) != 4:
        print("Usage: python analyze_retrieval_perplexity_final.py <log_file_path> <ppl_file_path> <output_file_path>")
        print("Example: python analyze_retrieval_perplexity_final.py log_0.00_1.00.log traj_abs_format_ppl.json retrieval_analysis.json")
        return
    
    log_file_path = sys.argv[1]
    ppl_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found: {log_file_path}")
        return
    
    if not os.path.exists(ppl_file_path):
        print(f"Error: Perplexity file not found: {ppl_file_path}")
        return
    
    # Run analysis
    analyze_log_file(log_file_path, ppl_file_path, output_file_path)


if __name__ == "__main__":
    main()
