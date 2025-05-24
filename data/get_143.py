import json

# 文件路径
file_184 = "/home/wiss/zhang/code/openeqa/3D-Mem/data/aeqa_questions-184.json"
file_41 = "/home/wiss/zhang/code/openeqa/3D-Mem/data/aeqa_questions-41.json"
output_file = "/home/wiss/zhang/code/openeqa/3D-Mem/data/aeqa_questions-143.json"

# 加载两个 JSON 文件
with open(file_184, "r") as f:
    questions_184 = json.load(f)

with open(file_41, "r") as f:
    questions_41 = json.load(f)

# 输出原始文件中 question 数量
print(f"原始文件包含 {len(questions_184)} 个问题")
print(f"需要排除的文件包含 {len(questions_41)} 个问题")

# 提取 41 个问题的 question_id
ids_41 = set(q["question_id"] for q in questions_41)

# 筛选不在 41 中的 question
filtered_questions = [q for q in questions_184 if q["question_id"] not in ids_41]

# 输出筛选后的数量
print(f"筛选后剩余 {len(filtered_questions)} 个问题")

# 写入输出文件
with open(output_file, "w") as f:
    json.dump(filtered_questions, f, indent=4)

print(f"已保存至 {output_file}")
