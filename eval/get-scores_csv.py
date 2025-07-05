import json
import numpy as np
import pickle
import os
import argparse
import pandas as pd

# 固定类别顺序
categories = [
    "object recognition",
    "spatial understanding",
    "object state recognition",
    "attribute recognition",
    "world knowledge",
    "object localization",
    "functional reasoning",
]

def spl(path_length, gt_path_length):
    return gt_path_length / max(gt_path_length, path_length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument("--dataset", default="open-eqa-184", type=str)
    args = parser.parse_args()

    data_path = args.result_path
    print(f"数据路径: {data_path}")
    matrics_path = os.path.dirname(data_path)  # 输出路径
    print(f"输出路径: {matrics_path}")
    os.makedirs(matrics_path, exist_ok=True)

    path_length_path = os.path.join(data_path, "path_length_list.pkl")
    gt_path = f"data/{args.dataset}.json"
    pred_path = os.path.join(data_path, "gpt_answer-metrics.json")
    baseline_path = f"data/{args.dataset}-gpt-4o-1234-metrics.json"
    gt_path_length_path = "data/gt_path_length.json"

    # 数据加载
    with open(gt_path_length_path, 'rb') as f:
        gt_path_length_map = json.load(f)
    with open(path_length_path, 'rb') as f:
        path_length_map = pickle.load(f)
    baseline_path_length_map = {k: float('inf') for k in gt_path_length_map}
    gt = json.load(open(gt_path))
    pred = json.load(open(pred_path))
    baseline = json.load(open(baseline_path))

    skip_scene_ids = []
    separate_scores = {}
    separate_spl = {}

    for question_id, score in baseline.items():
        question = next(q for q in gt if q['question_id'] == question_id)
        if question['episode_history'].split("-")[-1] in skip_scene_ids:
            continue

        gt_path_len = gt_path_length_map[question_id]
        if question_id not in pred:
            path_len = baseline_path_length_map[question_id]
        else:
            try:
                path_len = path_length_map[question_id]
                score = pred[question_id]
            except:
                path_len = baseline_path_length_map[question_id]

        category = question["category"]
        separate_scores.setdefault(category, []).append(score)
        separate_spl.setdefault(category, []).append(spl(path_len, gt_path_len))

    # 打印评估结果并保存为 DataFrame
    result = {"Method": os.path.basename(data_path)}
    total_scores = []
    total_spl_raw = []
    total_spl_weighted = []

    for cat in categories:
        scores = np.array(separate_scores.get(cat, []))
        spls = np.array(separate_spl.get(cat, []))
        llm_match = 100.0 * (scores - 1.0) / 4.0
        weighted = llm_match * spls

        result[f"{cat}_LLM_Match"] = np.mean(llm_match) if len(llm_match) else None
        result[f"{cat}_LLM_Match*SPL"] = np.mean(weighted) if len(weighted) else None

        total_scores.extend(scores)
        total_spl_raw.extend(spls)
        total_spl_weighted.extend(weighted)

        print(f"{cat}: {np.mean(llm_match):.2f}")
        print(f"{cat} SPL (raw): {np.mean(spls):.4f}")
        print(f"{cat} SPL (weighted): {np.mean(weighted):.2f}")
        print()

    result["LLM_Match"] = np.mean(100.0 * (np.array(total_scores) - 1.0) / 4.0)
    result["LLM_Match*SPL"] = np.mean(total_spl_weighted)
    result["SPL"] = np.mean(total_spl_raw)

    print(f"Total: {result['LLM_Match']:.2f}")
    print(f"Total SPL (raw): {result['SPL']:.4f}")
    print(f"Total SPL (weighted): {result['LLM_Match*SPL']:.2f}")

    # 保存为 CSV
    df = pd.DataFrame([result])
    out_path = os.path.join(matrics_path, f"{result['Method']}.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ 已保存 CSV 至: {out_path}")

if __name__ == "__main__":
    main()
