import json

file_184 = "data/aeqa_questions-184.json"
file_41 = "data/aeqa_questions-41.json"
output_file = "data/aeqa_questions-143.json"

with open(file_184, "r") as f:
    questions_184 = json.load(f)

with open(file_41, "r") as f:
    questions_41 = json.load(f)

print(f"")
print(f"")

ids_41 = set(q["question_id"] for q in questions_41)

filtered_questions = [q for q in questions_184 if q["question_id"] not in ids_41]

print(f"")

with open(output_file, "w") as f:
    json.dump(filtered_questions, f, indent=4)

print(f"")
