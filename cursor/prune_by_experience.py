import os
import json
import argparse
from collections import defaultdict


def load_exp(exp_path: str):
    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exp_steps = defaultdict(set)  # qid -> {step_keys}
    if isinstance(data, dict):
        for ep, qdict in data.items():
            if not isinstance(qdict, dict):
                continue
            for qid, qinfo in qdict.items():
                steps = (qinfo or {}).get('steps', {})
                if isinstance(steps, dict):
                    for sk in steps.keys():
                        exp_steps[qid].add(sk)
    return exp_steps


def prune_replay(replay_path: str, exp_steps: dict, out_path: str = None):
    with open(replay_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    removed_steps = 0
    removed_qids = 0
    kept_steps = 0

    if isinstance(data, dict):
        for ep, qdict in list(data.items()):
            if not isinstance(qdict, dict):
                continue
            for qid, qinfo in list(qdict.items()):
                steps = (qinfo or {}).get('steps', {})
                if not isinstance(steps, dict):
                    continue
                allowed = exp_steps.get(qid, set())
                # 删除不在 experience 中的 step
                for sk in list(steps.keys()):
                    if sk not in allowed:
                        steps.pop(sk, None)
                        removed_steps += 1
                    else:
                        kept_steps += 1
                # 若该 qid 没有任何 step，整个 qid 删掉
                if not steps:
                    qdict.pop(qid, None)
                    removed_qids += 1
                else:
                    qinfo['steps'] = steps
            # 若该 ep 为空也可选择删除（保留以防需要）

    save_to = out_path or (replay_path.replace('.json', '.pruned.json'))
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Replay] pruned file saved to: {save_to}")
    print(f"[Replay] removed_steps={removed_steps}, removed_qids={removed_qids}, kept_steps={kept_steps}")


def prune_index(index_path: str, exp_steps: dict, out_path: str = None):
    with open(index_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)

    removed = 0
    kept = 0

    if isinstance(idx, dict):
        for rel_key in list(idx.keys()):
            rec = idx.get(rel_key) or {}
            qid = rec.get('question_id')
            sk = rec.get('step_key')
            if not qid or not sk or sk not in exp_steps.get(qid, set()):
                idx.pop(rel_key, None)
                removed += 1
            else:
                kept += 1

    save_to = out_path or (index_path.replace('.json', '.pruned.json'))
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Index] pruned file saved to: {save_to}")
    print(f"[Index] removed_entries={removed}, kept_entries={kept}")


def load_index(index_path: str):
    with open(index_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    idx_qids = set()
    idx_steps = defaultdict(set)
    idx_none = defaultdict(int)
    if isinstance(idx, dict):
        for rel_key, rec in idx.items():
            if not isinstance(rec, dict):
                continue
            qid = rec.get('question_id')
            if not qid:
                parts = rel_key.split('/')
                if parts:
                    qid = parts[0]
            if not qid:
                continue
            idx_qids.add(qid)
            sk = rec.get('step_key')
            if sk:
                idx_steps[qid].add(sk)
            else:
                idx_none[qid] += 1
    return idx_qids, idx_steps, idx_none


def compare_and_report(exp_steps: dict, index_json_path: str):
    idx_qids, idx_steps, idx_none = load_index(index_json_path)
    exp_qids = set(exp_steps.keys())
    only_in_exp = sorted(exp_qids - idx_qids)
    only_in_idx = sorted(idx_qids - exp_qids)
    print(f"\n[Compare] exp_qids={len(exp_qids)} index_qids={len(idx_qids)}")
    if only_in_exp:
        print(f"[Compare] only_in_experience qids ({len(only_in_exp)}):")
        for q in only_in_exp[:50]:
            print('  ', q)
        if len(only_in_exp) > 50:
            print(f"  ... (+{len(only_in_exp)-50} more)")
    if only_in_idx:
        print(f"[Compare] only_in_index qids ({len(only_in_idx)}):")
        for q in only_in_idx[:50]:
            print('  ', q)
        if len(only_in_idx) > 50:
            print(f"  ... (+{len(only_in_idx)-50} more)")

    common = sorted(exp_qids & idx_qids)
    mism = 0
    for qid in common:
        exp_set = exp_steps.get(qid, set())
        idx_set = idx_steps.get(qid, set())
        miss = sorted(exp_set - idx_set)
        extra = sorted(idx_set - exp_set)
        if miss or extra:
            mism += 1
            print(f"[Compare] MISMATCH qid={qid}")
            if miss:
                print("  steps in exp but NOT in index:", ', '.join(miss))
            if extra:
                print("  steps in index but NOT in exp:", ', '.join(extra))
            n_none = idx_none.get(qid, 0)
            if n_none:
                print(f"  note: {n_none} entries have step_key=None")
    if mism == 0 and not only_in_exp and not only_in_idx:
        print("[Compare] OK: question_id 和 step_key 完全对齐")


def prune_experience_by_index(exp_path: str, idx_steps: dict, out_path: str = None):
    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    removed_steps = 0
    removed_qids = 0
    kept_steps = 0

    if isinstance(data, dict):
        for ep, qdict in list(data.items()):
            if not isinstance(qdict, dict):
                continue
            for qid, qinfo in list(qdict.items()):
                steps = (qinfo or {}).get('steps', {})
                if not isinstance(steps, dict):
                    continue
                allowed = idx_steps.get(qid, set())
                # 删除不在 index 中的 step
                for sk in list(steps.keys()):
                    if sk not in allowed:
                        steps.pop(sk, None)
                        removed_steps += 1
                    else:
                        kept_steps += 1
                if not steps:
                    qdict.pop(qid, None)
                    removed_qids += 1
                else:
                    qinfo['steps'] = steps

    save_to = out_path or (exp_path.replace('.json', '.pruned.json'))
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Exp] pruned file saved to: {save_to}")
    print(f"[Exp] removed_steps={removed_steps}, removed_qids={removed_qids}, kept_steps={kept_steps}")


# ===== Mutual prune utils =====
def build_intersection_steps(exp_steps: dict, idx_steps: dict):
    allowed = {}
    common_qids = set(exp_steps.keys()) & set(idx_steps.keys())
    for qid in common_qids:
        inter = set(exp_steps.get(qid, set())) & set(idx_steps.get(qid, set()))
        if inter:
            allowed[qid] = inter
    return allowed


def prune_experience_with_allowed(exp_path: str, allowed_steps: dict, out_path: str = None):
    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    removed_steps = removed_qids = kept_steps = 0
    if isinstance(data, dict):
        for ep, qdict in list(data.items()):
            if not isinstance(qdict, dict):
                continue
            for qid, qinfo in list(qdict.items()):
                if qid not in allowed_steps:
                    qdict.pop(qid, None)
                    removed_qids += 1
                    continue
                steps = (qinfo or {}).get('steps', {})
                if not isinstance(steps, dict):
                    qdict.pop(qid, None)
                    removed_qids += 1
                    continue
                allow = allowed_steps[qid]
                for sk in list(steps.keys()):
                    if sk not in allow:
                        steps.pop(sk, None)
                        removed_steps += 1
                    else:
                        kept_steps += 1
                if not steps:
                    qdict.pop(qid, None)
                    removed_qids += 1
                else:
                    qinfo['steps'] = steps
    save_to = out_path or exp_path
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Exp∩] saved: {save_to} | removed_steps={removed_steps}, removed_qids={removed_qids}, kept_steps={kept_steps}")


def prune_index_with_allowed(index_path: str, allowed_steps: dict, out_path: str = None):
    with open(index_path, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    removed = kept = 0
    if isinstance(idx, dict):
        for rel_key in list(idx.keys()):
            rec = idx.get(rel_key) or {}
            qid = rec.get('question_id')
            sk = rec.get('step_key')
            if not qid or not sk or qid not in allowed_steps or sk not in allowed_steps[qid]:
                idx.pop(rel_key, None)
                removed += 1
            else:
                kept += 1
    save_to = out_path or index_path
    with open(save_to, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Index∩] saved: {save_to} | removed_entries={removed}, kept_entries={kept}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help='path to experience_output.json')
    parser.add_argument('--replay', required=False, help='path to replay_step_info.json')
    parser.add_argument('--index', required=False, help='path to .frontier_ahash_index.json')
    parser.add_argument('--out_replay', required=False, default='')
    parser.add_argument('--out_index', required=False, default='')
    parser.add_argument('--prune_exp_by_index', action='store_true', help='also prune experience_output.json by index steps')
    parser.add_argument('--out_exp', required=False, default='')
    parser.add_argument('--mutual', action='store_true', help='mutually prune: keep only intersection of (exp steps) ∩ (index steps) in both files and compare')
    args = parser.parse_args()

    exp_steps = load_exp(args.exp)
    print(f"[EXP] loaded qids={len(exp_steps)}")

    if args.replay:
        prune_replay(args.replay, exp_steps, args.out_replay or None)
        # 对裁剪结果再次比较
        cmp_path = (args.out_replay or args.replay.replace('.json', '.pruned.json'))
        print("\n[Compare after prune: replay]")
        # replay 的结构复杂，这里仅提示已保存；如需更细比较可复用 compare_experience_and_index.py

    if args.index:
        prune_index(args.index, exp_steps, args.out_index or None)
        cmp_path = (args.out_index or args.index.replace('.json', '.pruned.json'))
        print("\n[Compare after prune: index]")
        compare_and_report(exp_steps, cmp_path)

    # 反向：用 index 裁剪 experience，并对比
    if args.prune_exp_by_index and args.index:
        _, idx_steps, _ = load_index(args.index)
        prune_experience_by_index(args.exp, idx_steps, args.out_exp or None)
        exp_cmp = (args.out_exp or args.exp.replace('.json', '.pruned.json'))
        print("\n[Compare after prune: experience]")
        compare_and_report(load_exp(exp_cmp), args.index)

    # 互删：仅保留交集
    if args.mutual and args.index:
        _, idx_steps, _ = load_index(args.index)
        allowed = build_intersection_steps(exp_steps, idx_steps)
        # 覆盖保存（或到指定路径）
        out_exp_path = args.out_exp or args.exp
        out_idx_path = args.out_index or args.index
        prune_experience_with_allowed(args.exp, allowed, out_exp_path)
        prune_index_with_allowed(args.index, allowed, out_idx_path)
        print("\n[Compare after mutual prune]")
        compare_and_report(load_exp(out_exp_path), out_idx_path)


if __name__ == '__main__':
    main()


