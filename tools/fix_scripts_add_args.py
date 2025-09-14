import os

ROOT = "/home/hpc/v100dd/v100dd12/code/3D-Mem"
SCRIPTS_DIR = os.path.join(ROOT, "script/experience")
RETRIEVE_ROOT = "/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168"
EXP_TUPLE = f"{RETRIEVE_ROOT}/exp_tuple_v0.json"


def process_file(path: str) -> bool:
    changed = False
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "run_aeqa_evaluation_qwen.py" in line and "python" in line:
            new_line = line
            # normalize retrieve_root if present
            if "--retrieve_root" in new_line:
                parts = new_line.split()
                for j, p in enumerate(parts):
                    if p == "--retrieve_root" and j + 1 < len(parts):
                        parts[j + 1] = RETRIEVE_ROOT
                        new_line = " ".join(parts)
                        break
            # add retrieve_root if missing
            if "--retrieve_root" not in new_line:
                new_line = new_line.rstrip("\n") + f"     --retrieve_root {RETRIEVE_ROOT}\n"
            # normalize exp_tuple if present
            if "--exp_tuple" in new_line:
                parts = new_line.split()
                for j, p in enumerate(parts):
                    if p == "--exp_tuple" and j + 1 < len(parts):
                        parts[j + 1] = EXP_TUPLE
                        new_line = " ".join(parts)
                        break
            # add exp_tuple if missing
            if "--exp_tuple" not in new_line:
                new_line = new_line.rstrip("\n") + f"     --exp_tuple {EXP_TUPLE}\n"

            if new_line != line:
                lines[i] = new_line
                changed = True
            break
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def main():
    updated = 0
    for root, _, files in os.walk(SCRIPTS_DIR):
        for fn in files:
            if not fn.endswith(".sh"):
                continue
            fp = os.path.join(root, fn)
            if process_file(fp):
                updated += 1
    print(f"Updated {updated} scripts.")


if __name__ == "__main__":
    main()


