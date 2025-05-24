import json
import re

json_path = "/home/wiss/zhang/code/openeqa/3D-Mem/data/aeqa_questions-184.json"
out_path = "/home/wiss/zhang/code/openeqa/3D-Mem/slurm/184-aeqa-eval-68361.out"
output_path = "/home/wiss/zhang/projects/openeqa/aeqa/results_184/184_sum.json"

def parse_out_file(out_path):
    with open(out_path, "r") as f:
        lines = f.readlines()

    result = {}
    fail_reasons = {}
    for i, line in enumerate(lines):
        # 匹配失败
        fail_match = re.search(r"Question id (\S+) failed", line)
        if fail_match:
            qid = fail_match.group(1)
            reason = None
            # 前一行是否有具体错误原因
            if i > 0 and "invalid:" in lines[i - 1]:
                reason_match = re.search(r"Question id \S+ invalid: (.+)", lines[i - 1])
                if reason_match:
                    reason = reason_match.group(1)
            result[qid] = "failed"
            fail_reasons[qid] = reason or "step limit exceeded"
        # 匹配成功
        success_match = re.search(r"Question id (\S+) finish successfully", line)
        if success_match:
            qid = success_match.group(1)
            result[qid] = "success"
    return result, fail_reasons

def analyze(json_path, out_path, output_path):
    with open(json_path, "r") as f:
        questions = json.load(f)

    status_dict, reason_dict = parse_out_file(out_path)

    summary = {
        "total": len(questions),
        "success": 0,
        "failures": {},
        "details": []
    }

    for q in questions:
        qid = q["question_id"]
        q_entry = {
            "question_id": qid,
            "question": q["question"],
            "answer": q["answer"],
            "status": status_dict.get(qid, "not_found")
        }
        if status_dict.get(qid) == "failed":
            reason = reason_dict.get(qid, "unknown")
            q_entry["fail_reason"] = reason
            summary["failures"][reason] = summary["failures"].get(reason, 0) + 1
        elif status_dict.get(qid) == "success":
            summary["success"] += 1

        summary["details"].append(q_entry)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    analyze(json_path, out_path, output_path)
