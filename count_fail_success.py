import json
import re

json_path = "/home/wiss/zhang/code/openeqa/3D-Mem/data/aeqa_questions-41.json"
out_path = "/home/wiss/zhang/code/openeqa/3D-Mem/slurm/pred/pred-41-69279.out"
output_path = "/home/wiss/zhang/projects/openeqa/aeqa/results_41_pred/41_pred_sum.json"

def parse_out_file(out_path):
    with open(out_path, "r") as f:
        lines = f.readlines()

    result = {}
    fail_reasons = {}
    step_counts = {}

    current_qid = None
    step_pattern = re.compile(r"== step: (\d+)")
    
    for i, line in enumerate(lines):
        step_match = step_pattern.search(line)
        if step_match and current_qid is not None:
            step_num = int(step_match.group(1))
            if current_qid not in step_counts:
                step_counts[current_qid] = set()
            step_counts[current_qid].add(step_num)

        # 失败匹配
        fail_match = re.search(r"Question id (\S+) failed", line)
        if fail_match:
            qid = fail_match.group(1)
            current_qid = qid
            reason = None
            if i > 0 and "invalid:" in lines[i - 1]:
                reason_match = re.search(r"Question id \S+ invalid: (.+)", lines[i - 1])
                if reason_match:
                    reason = reason_match.group(1)
            result[qid] = "failed"
            fail_reasons[qid] = reason or "step limit exceeded"

        # 成功匹配
        success_match = re.search(r"Question id (\S+) finish successfully", line)
        if success_match:
            qid = success_match.group(1)
            current_qid = qid
            result[qid] = "success"

    # 转换 set 为 step 数量
    step_counts = {qid: len(steps) for qid, steps in step_counts.items()}
    return result, fail_reasons, step_counts

def analyze(json_path, out_path, output_path):
    with open(json_path, "r") as f:
        questions = json.load(f)

    status_dict, reason_dict, step_counts = parse_out_file(out_path)

    summary = {
        "total": len(questions),
        "success": 0,
        "failures": {},
        "details": []
    }

    for q in questions:
        qid = q["question_id"]
        status = status_dict.get(qid, "not_found")
        steps = step_counts.get(qid, 0)

        q_entry = {
            "question_id": qid,
            "question": q["question"],
            "answer": q["answer"],
            "status": status,
            "steps": steps
        }

        if status == "failed":
            reason = reason_dict.get(qid, "unknown")
            q_entry["fail_reason"] = reason

            if reason not in summary["failures"]:
                summary["failures"][reason] = {"count": 0, "question_ids": []}
            summary["failures"][reason]["count"] += 1
            summary["failures"][reason]["question_ids"].append(qid)
        elif status == "success":
            summary["success"] += 1

        summary["details"].append(q_entry)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    analyze(json_path, out_path, output_path)
