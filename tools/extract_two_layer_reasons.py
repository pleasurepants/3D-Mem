#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\s*-\s*")
QUESTION_RE = re.compile(r"Question id ([0-9a-fA-F-]+) initialization successful")
STEP_RE = re.compile(r"== step:\s*(\d+)")
LAYER0_RE = re.compile(r"reason for layer0 selection:", re.IGNORECASE)
LAYER1_RE = re.compile(r"Reason:\s*(.*)", re.IGNORECASE)
INPUT_PATH = Path(
    "/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/format_unformat/unformat/gpt/sim/top-5/unformat-all-sim-t5-s568-3141243.out"
)
QUESTIONS_PATH = Path("/home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-41.json")
OUTPUT_PATH = Path("/home/hpc/v100dd/v100dd12/code/3D-Mem/teaser/reexplore.json")


def strip_timestamp(line: str) -> str:
    return TIMESTAMP_RE.sub("", line, count=1)


def parse_file(path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    result: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    current_qid: Optional[str] = None
    current_step: Optional[int] = None

    capturing_layer0 = False
    layer0_lines: List[str] = []

    capturing_layer1 = False
    layer1_lines: List[str] = []
    layer1_depth = 0

    def ensure_entry() -> Optional[Dict[str, List[str]]]:
        if current_qid is None or current_step is None:
            return None
        step_dict = result.setdefault(current_qid, {})
        key = str(current_step)
        entry = step_dict.setdefault(key, {"layer0": [], "layer1": []})
        return entry

    def finalize_layer0():
        nonlocal capturing_layer0, layer0_lines
        if not capturing_layer0:
            return
        capturing_layer0 = False
        text = "\n".join(layer0_lines).strip()
        layer0_lines = []
        if text:
            entry = ensure_entry()
            if entry is not None:
                entry["layer0"].append(text)

    def finalize_layer1():
        nonlocal capturing_layer1, layer1_lines, layer1_depth
        if not capturing_layer1:
            return
        capturing_layer1 = False
        layer1_depth = 0
        if layer1_lines:
            # Remove trailing closing bracket if present
            if layer1_lines[-1].strip().endswith("]"):
                layer1_lines[-1] = layer1_lines[-1].rstrip().rstrip("]")
            cleaned = [line.rstrip() for line in layer1_lines if line.strip()]
            text = "\n".join(cleaned).strip()
        else:
            text = ""
        layer1_lines = []
        if text:
            entry = ensure_entry()
            if entry is not None:
                entry["layer1"].append(text)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            has_timestamp = bool(TIMESTAMP_RE.match(line))
            content = strip_timestamp(line) if has_timestamp else line

            if capturing_layer0 and has_timestamp:
                finalize_layer0()

            if capturing_layer1:
                layer1_lines.append(content)
                layer1_depth += content.count("[") - content.count("]")
                if layer1_depth <= 0:
                    finalize_layer1()
                continue

            q_match = QUESTION_RE.search(content)
            if q_match:
                finalize_layer0()
                finalize_layer1()
                current_qid = q_match.group(1)
                current_step = None
                continue

            step_match = STEP_RE.search(content)
            if step_match:
                finalize_layer0()
                finalize_layer1()
                current_step = int(step_match.group(1))
                continue

            if LAYER0_RE.search(content):
                finalize_layer0()
                capturing_layer0 = True
                layer0_lines = []
                after_colon = content.split(":", 1)[-1].strip()
                if after_colon:
                    layer0_lines.append(after_colon)
                continue

            reason_match = LAYER1_RE.match(content)
            if reason_match:
                finalize_layer1()
                start_text = reason_match.group(1).lstrip()
                capturing_layer1 = True
                layer1_lines = []
                if start_text.startswith("["):
                    layer1_depth = 1 + start_text.count("[", 1) - start_text.count("]")
                    start_text = start_text[1:]
                else:
                    layer1_depth = start_text.count("[") - start_text.count("]")
                if start_text.strip():
                    layer1_lines.append(start_text)
                if layer1_depth <= 0:
                    finalize_layer1()
                continue

            if capturing_layer0 and not has_timestamp:
                layer0_lines.append(content)

    finalize_layer0()
    finalize_layer1()
    return result


def load_question_meta(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return {}
    meta: Dict[str, Dict[str, Optional[str]]] = {}
    for item in data:
        qid = item.get("question_id")
        if not qid:
            continue
        meta[qid] = {
            "question": item.get("question"),
            "answer": item.get("answer"),
        }
    return meta


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_PATH}")

    question_meta = load_question_meta(QUESTIONS_PATH)
    parsed = parse_file(INPUT_PATH)
    final_output = {}
    for qid, steps in parsed.items():
        meta = question_meta.get(qid, {})
        final_output[qid] = {
            "question": meta.get("question"),
            "answer": meta.get("answer"),
            "steps": steps,
        }
    json_str = json.dumps(final_output, ensure_ascii=False, indent=2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json_str, encoding="utf-8")


if __name__ == "__main__":
    main()

