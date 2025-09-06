import os
import sys
import json
import argparse
from collections import defaultdict


def load_experience_steps(exp_path: str):
    """
    返回：
      - exp_qids: set[str]
      - exp_steps: dict[qid] -> set[step_key]
    说明：experience_output.json 结构为 episode_id -> question_id -> {"steps": {...}}
    """
    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    exp_qids = set()
    exp_steps = defaultdict(set)

    if not isinstance(data, dict):
        return exp_qids, exp_steps

    for ep_id, qdict in data.items():
        if not isinstance(qdict, dict):
            continue
        for qid, qinfo in qdict.items():
            exp_qids.add(qid)
            if isinstance(qinfo, dict):
                steps = qinfo.get('steps', {})
                if isinstance(steps, dict):
                    for step_key in steps.keys():
                        exp_steps[qid].add(step_key)
    return exp_qids, exp_steps


def load_index_steps(index_path: str):
    """
    返回：
      - idx_qids: set[str]
      - idx_steps: dict[qid] -> set[step_key]
      - idx_none_steps: dict[qid] -> count of entries with step_key=None (用于提示解析失败情况)
    说明：.frontier_ahash_index.json 的键是 "qid/frontier/xxx.png"；值里包含
          question_id, filename, step_key, level, bits
    """
    with open(index_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)

    idx_qids = set()
    idx_steps = defaultdict(set)
    idx_none_steps = defaultdict(int)

    if not isinstance(idx, dict):
        return idx_qids, idx_steps, idx_none_steps

    for rel_key, rec in idx.items():
        if not isinstance(rec, dict):
            continue
        qid = rec.get('question_id')
        step_key = rec.get('step_key')
        if not qid:
            # 尝试从键解析 qid（键形如 qid/frontier/xxx.png）
            parts = rel_key.split('/')
            if parts:
                qid = parts[0]
        if not qid:
            # 无法定位 qid，跳过
            continue
        idx_qids.add(qid)
        if step_key:
            idx_steps[qid].add(step_key)
        else:
            idx_none_steps[qid] += 1
    return idx_qids, idx_steps, idx_none_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=False, default='/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/experience_output.json')
    parser.add_argument('--index', required=False, default='/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/.frontier_ahash_index.json')
    args = parser.parse_args()

    exp_qids, exp_steps = load_experience_steps(args.exp)
    idx_qids, idx_steps, idx_none_steps = load_index_steps(args.index)

    print(f'[Summary] exp_qids={len(exp_qids)}, index_qids={len(idx_qids)}')

    # 1) question_id 对齐检查
    only_in_exp = sorted(exp_qids - idx_qids)
    only_in_idx = sorted(idx_qids - exp_qids)

    if only_in_exp:
        print('\n[Only in experience_output.json] question_id count=', len(only_in_exp))
        for q in only_in_exp[:50]:
            print('  ', q)
        if len(only_in_exp) > 50:
            print(f'  ... (+{len(only_in_exp)-50} more)')

    if only_in_idx:
        print('\n[Only in .frontier_ahash_index.json] question_id count=', len(only_in_idx))
        for q in only_in_idx[:50]:
            print('  ', q)
        if len(only_in_idx) > 50:
            print(f'  ... (+{len(only_in_idx)-50} more)')

    # 2) 每个 question_id 的 step_key 集合对齐检查
    common_qids = sorted(exp_qids & idx_qids)
    mismatch_count = 0
    for qid in common_qids:
        exp_set = exp_steps.get(qid, set())
        idx_set = idx_steps.get(qid, set())
        miss_in_idx = sorted(exp_set - idx_set)
        extra_in_idx = sorted(idx_set - exp_set)
        none_cnt = idx_none_steps.get(qid, 0)

        if miss_in_idx or extra_in_idx:
            mismatch_count += 1
            print(f"\n[MISMATCH] question_id={qid}")
            if miss_in_idx:
                print('  steps in exp but NOT in index:', ', '.join(miss_in_idx))
            if extra_in_idx:
                print('  steps in index but NOT in exp:', ', '.join(extra_in_idx))
            if none_cnt:
                print(f'  note: {none_cnt} index entries had step_key=None (filename may not follow pattern)')

    if mismatch_count == 0 and not only_in_exp and not only_in_idx:
        print('\n[OK] question_id 和 step_key 集合完全一一对应。')
    else:
        print(f"\n[Done] mismatched_qids={mismatch_count}, only_in_exp={len(only_in_exp)}, only_in_index={len(only_in_idx)}")


if __name__ == '__main__':
    main()


