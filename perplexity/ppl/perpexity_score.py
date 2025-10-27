#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate perplexity scores for the abstraction field in JSON files
Using Hugging Face language models to compute perplexity
"""

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
import os


def calculate_perplexity(text, model, tokenizer, device, max_length=1024):
    """
    Calculate perplexity score for given text
    
    Args:
        text: Input text
        model: Language model
        tokenizer: Tokenizer
        device: Computing device
        max_length: Maximum sequence length
    
    Returns:
        Perplexity score
    """
    # Encode text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    
    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity


def calculate_perplexity_sliding_window(text, model, tokenizer, device, max_length=1024, stride=512):
    """
    Calculate perplexity for long texts using sliding window method
    Reference: https://huggingface.co/docs/transformers/perplexity
    
    Args:
        text: Input text
        model: Language model
        tokenizer: Tokenizer
        device: Computing device
        max_length: Maximum sequence length
        stride: Sliding window stride
    
    Returns:
        Perplexity score
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    
    # If text is short, calculate directly
    if seq_len <= max_length:
        return calculate_perplexity(text, model, tokenizer, device, max_length)
    
    # Use sliding window method (reference: Hugging Face official documentation)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # May differ from stride (last window)
        
        input_ids_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            
            # Loss is calculated using CrossEntropyLoss which averages over valid labels
            # Note: model only calculates loss over trg_len - 1 labels due to internal label shift
            neg_log_likelihood = outputs.loss
        
        # Accumulate total negative log-likelihood and total token count
        num_valid_tokens = (target_ids != -100).sum().item()  # Number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # Subtract batch_size due to internal label shift
        
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
        
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    # Calculate average negative log-likelihood and perplexity
    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl.item()


def process_json_file(json_path, model, tokenizer, device, output_path=None, use_sliding_window=True):
    """
    Process JSON file and calculate perplexity for each entry's abstraction field
    
    Args:
        json_path: Path to JSON file
        model: Language model
        tokenizer: Tokenizer
        device: Computing device
        output_path: Path to output results file
        use_sliding_window: Whether to use sliding window method
    """
    # Read JSON file
    print(f"Reading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} entries")
    
    # Calculate perplexity for each entry
    results = {}
    perplexity_scores = []
    
    for key, value in tqdm(data.items(), desc="Calculating perplexity"):
        if 'abstraction' not in value:
            print(f"Warning: Entry {key} has no abstraction field, skipping")
            continue
        
        abstraction_text = value['abstraction']
        
        # Calculate perplexity
        if use_sliding_window:
            ppl = calculate_perplexity_sliding_window(abstraction_text, model, tokenizer, device)
        else:
            ppl = calculate_perplexity(abstraction_text, model, tokenizer, device)
        
        # Simplified output format: each question-id maps to abstraction and perplexity score
        results[key] = {
            'abstraction': abstraction_text,
            'perplexity': ppl
        }
        perplexity_scores.append(ppl)
        
        # Print partial results in real-time
        if len(results) % 10 == 0:
            print(f"\nProcessed {len(results)} entries")
            print(f"Latest entry {key}: perplexity = {ppl:.2f}")
    
    # Calculate statistics
    perplexity_scores = np.array(perplexity_scores)
    stats = {
        'mean': float(np.mean(perplexity_scores)),
        'median': float(np.median(perplexity_scores)),
        'std': float(np.std(perplexity_scores)),
        'min': float(np.min(perplexity_scores)),
        'max': float(np.max(perplexity_scores)),
        'total_entries': len(results)
    }
    
    # Output statistics
    print("\n" + "="*50)
    print("Perplexity Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print("="*50)
    
    # Save results (including statistics)
    if output_path is None:
        output_path = json_path.replace('.json', '_perplexity.json')
    
    output_data = {
        'statistics': stats,
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return results, stats


def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity scores for abstraction field in JSON files')
    parser.add_argument('--json_dir', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output results file (default: input_filename_perplexity.json)')
    parser.add_argument('--model_name', type=str, default='/anvme/workspace/v100dd12-3dmem/model/Qwen2.5-7B',
                        help='Model name or path to use (default: Qwen2.5-7B)')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024,
                        help='Sliding window stride (default: 1024)')
    parser.add_argument('--no_sliding_window', action='store_true',
                        help='Do not use sliding window method (may truncate long texts)')
    parser.add_argument('--device', type=str, default=None,
                        help='Computing device (default: auto-select cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        print("Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # If model is already assigned to device via device_map="auto", no need to call .to(device) again
        device = next(model.parameters()).device
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Trying to load with default settings...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        model.eval()
        print("Model loaded successfully (default settings)")
    
    # Process JSON file
    process_json_file(
        json_path=args.json_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_path=args.output_dir,
        use_sliding_window=not args.no_sliding_window
    )


if __name__ == '__main__':
    main()

