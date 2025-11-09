import json
import argparse
import os
import sys
import time
import logging
import openai
from openai import OpenAI
from typing import Optional

# 添加工程根目录到路径以导入相关模块
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.const_gpt import END_POINT, OPENAI_KEY

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


def format_content(contents):
    """Format content for OpenAI API"""
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formated_content


def call_openai_api(sys_prompt, contents, seed: Optional[int] = None) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            # 支持从参数或环境变量注入 seed（优先参数，其次 VLLM_SEED）
            # 读取优先级：参数 seed > cfg.chat_seed（经外层传入）> 环境变量 VLLM_SEED
            _seed_env = None
            try:
                _seed_env = int(os.getenv("VLLM_SEED")) if os.getenv("VLLM_SEED") is not None else None
            except Exception:
                _seed_env = None
            _seed = seed if seed is not None else _seed_env
            try:
                logging.info(f"[ChatSeed] using seed={_seed}")
            except Exception:
                pass
            completion = client.chat.completions.create(
                model="gpt-4o",  # gpt-4o-internvl-minicpm-qwen
                messages=message_text,
                temperature=0.7,
                max_tokens=4096, # 4096 for gpt-4o
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                **({"seed": int(_seed)} if _seed is not None else {}),
            )
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None


def load_questions(questions_path):
    """Load questions data"""
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Build question_id -> question mapping
    question_map = {}
    for item in questions_data:
        question_map[item['question_id']] = {
            'question': item['question'],
            'answer': item.get('answer', ''),
            'category': item.get('category', ''),
        }
    
    return question_map


def evaluate_abstraction(abstraction, question_info, dimension):
    """
    Evaluate abstraction score on a specific dimension
    
    Args:
        abstraction: The abstraction text to evaluate
        question_info: Dictionary containing question, answer, and category information
        dimension: Evaluation dimension ('generality', 'relevance', 'conciseness', 'actionability')
    
    Returns:
        Dictionary with score and reason
    """
    
    # System prompt
    sys_prompt = """You are an expert evaluator for navigation task experiences.

Task: Evaluate the given experience on a specific dimension and provide a score from 1 to 5.

Universal Scoring Guide:
- Score 5: Excellent - meets all criteria with high quality
- Score 4: Good - meets most criteria with minor issues
- Score 3: Fair - meets some criteria but has notable limitations
- Score 2: Poor - meets few criteria with significant issues
- Score 1: Very Poor - fails to meet the criteria

Chain-of-Thought Evaluation Process:
Step 1: Carefully read and understand the experience content
Step 2: Identify the key aspects relevant to the evaluation dimension
Step 3: Analyze how well the experience meets the criteria
Step 4: Weigh the strengths against the weaknesses
Step 5: Map your analysis to the scoring guide (1-5)
Step 6: Formulate a clear explanation for your score

Requirements:
- Score must be an integer between 1 and 5
- Provide reasoning in 2-3 sentences that references specific evidence
- Be objective and consistent with the criteria

Output Format (strictly follow - reason BEFORE score):
{
    "reason": "<your detailed explanation>",
    "score": <integer 1-5>
}"""

    # Build different content prompts based on dimensions
    if dimension == "generality":
        content = f"""Evaluation Dimension: GENERALITY

Definition: Measures whether the experience describes generalizable patterns and strategies that can transfer to different but similar scenarios, rather than being overly specific to particular instances or environments.

Experience to Evaluate:
{abstraction}"""

    elif dimension == "relevance":
        content = f"""Evaluation Dimension: RELEVANCE

Question Context:
- Question: {question_info['question']}
- Answer: {question_info['answer']}
- Category: {question_info['category']}

Definition: Measures how well the experience directly addresses the question topic and provides information that supports or explains the answer, without including unrelated content.

Experience to Evaluate:
{abstraction}"""

    elif dimension == "conciseness":
        content = f"""Evaluation Dimension: CONCISENESS

Definition: Measures how efficiently the experience conveys information - whether it expresses ideas clearly and completely without unnecessary repetition, verbosity, or redundant phrases.

Experience to Evaluate:
{abstraction}"""

    elif dimension == "actionability":
        content = f"""Evaluation Dimension: ACTIONABILITY

Definition: Measures how well the experience provides concrete and executable guidance that can be directly applied in practice, with clear steps or strategies rather than vague observations.

Experience to Evaluate:
{abstraction}"""

    else:
        raise ValueError(f"Unknown dimension: {dimension}")
    
    # Call API
    response = call_openai_api(sys_prompt, [(content,)])
    
    if response is None:
        return {"reason": "API call failed", "score": 0}
    
    # Parse response
    try:
        # Try to extract JSON from response (support both "reason" first and "score" first)
        import re
        json_match = re.search(r'\{[^{}]*"(reason|score)"[^{}]*"(reason|score)"[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            # If no JSON format found, try direct parsing
            result = json.loads(response)
            return result
    except Exception as e:
        print(f"Failed to parse response: {e}")
        print(f"Original response: {response}")
        return {"reason": f"Parsing failed: {str(e)}", "score": 0}


def main():
    parser = argparse.ArgumentParser(description='Multi-dimensional scoring for experience abstractions')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Input JSON file path containing question_id and abstraction')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--questions_json', type=str,
                        default='/home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-184.json',
                        help='Questions data JSON file path')
    
    args = parser.parse_args()
    
    print(f"Loading input file: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    print(f"Loading questions file: {args.questions_json}")
    question_map = load_questions(args.questions_json)
    
    # Get results dictionary
    if "results" in input_data:
        results = input_data["results"]
    else:
        results = input_data
    
    total_questions = len(results)
    print(f"Total {total_questions} questions to evaluate")
    
    # Evaluation dimensions
    dimensions = ["generality", "relevance", "conciseness", "actionability"]
    dimension_names = {
        "generality": "Generality",
        "relevance": "Relevance",
        "conciseness": "Conciseness",
        "actionability": "Actionability"
    }
    
    # Iterate through each question_id
    for idx, (question_id, data) in enumerate(results.items(), 1):
        print(f"\nProcessing [{idx}/{total_questions}] {question_id}")
        
        abstraction = data.get("abstraction", "")
        if not abstraction:
            print(f"  Warning: question_id {question_id} has no abstraction, skipping")
            continue
        
        # Get question information
        question_info = question_map.get(question_id)
        if not question_info:
            print(f"  Warning: question not found for question_id {question_id}")
            question_info = {
                'question': 'Unknown question',
                'answer': 'Unknown',
                'category': 'Unknown',
            }
        
        # Add question information to results
        data['question'] = question_info['question']
        data['answer'] = question_info['answer']
        data['category'] = question_info['category']
        
        # Initialize scores dictionary
        if "scores" not in data:
            data["scores"] = {}
        
        # Evaluate each dimension
        for dimension in dimensions:
            dim_name = dimension_names[dimension]
            print(f"  Evaluating {dim_name}...")
            
            # Check if already scored
            if dimension in data["scores"]:
                print(f"    Score already exists, skipping")
                continue
            
            score_result = evaluate_abstraction(abstraction, question_info, dimension)
            data["scores"][dimension] = score_result
            
            print(f"    Score: {score_result.get('score', 0)}/5")
            print(f"    Reason: {score_result.get('reason', 'N/A')[:50]}...")
            
            # Avoid too frequent requests
            time.sleep(1)
        
        # Calculate average score
        valid_scores = [data["scores"][dim].get("score", 0) for dim in dimensions 
                       if dim in data["scores"] and data["scores"][dim].get("score", 0) > 0]
        
        if valid_scores:
            data["scores"]["average"] = {
                "score": round(sum(valid_scores) / len(valid_scores), 2),
                "reason": f"Average of {len(valid_scores)} dimensions"
            }
            print(f"  Average Score: {data['scores']['average']['score']}/5")
        else:
            data["scores"]["average"] = {
                "score": 0,
                "reason": "No valid scores available"
            }
    
    # Save results
    output_data = input_data.copy()
    output_data["results"] = results
    
    print(f"\nSaving results to: {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Scoring completed!")


if __name__ == "__main__":
    main()

