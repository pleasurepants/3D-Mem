#!/bin/bash
set -euo pipefail

# Config
ROOT="/home/hpc/v100dd/v100dd12/code/3D-Mem"
SLURM_OUT_DIR="$ROOT/slurm/ee"
mkdir -p "$SLURM_OUT_DIR"

declare -A NAME_TO_SCRIPT=(
  ["q-ee-random-top1"]="$ROOT/script/alex/alex_qwen.sh"
  ["q-ee-random-top1-19"]="$ROOT/script/alex/alex_qwen_seed19.sh"
  ["q-ee-random-top1-82"]="$ROOT/script/alex/alex_qwen_seed82.sh"
  ["q-ee-random-top1-568"]="$ROOT/script/alex/alex_qwen_seed568.sh"
)

NAMES=("q-ee-random-top1" "q-ee-random-top1-19" "q-ee-random-top1-82" "q-ee-random-top1-568")

STATE_DIR="$ROOT/cursor/.monitor_state"
mkdir -p "$STATE_DIR"

log() { echo "[$(date +'%F %T')] $*"; }

get_jobid_by_name() {
  local name="$1"
  squeue -u "$USER" -o "%i %j %t" | awk -v n="$name" '$2==n {print $1}' | head -n1
}

get_state_by_jobid() {
  local jid="$1"
  sacct -j "$jid" -X -n -o State | head -n1 | awk '{print $1}'
}

tail_log_for_jobid() {
  local jid="$1"
  local f
  f=$(ls -t "$SLURM_OUT_DIR"/*"$jid"*.out 2>/dev/null | head -n1 || true)
  if [[ -n "$f" && -f "$f" ]]; then
    log "--- tail -n 100 $f ---"
    tail -n 100 "$f" || true
  else
    log "No slurm out file found for JobID=$jid under $SLURM_OUT_DIR"
  fi
}

resubmit_by_name() {
  local name="$1"
  local script="${NAME_TO_SCRIPT[$name]}"
  if [[ -z "$script" || ! -f "$script" ]]; then
    log "No script mapped for $name or file missing: $script"
    return 1
  fi
  local out
  out=$(sbatch "$script")
  log "Resubmitted $name via $script => $out"
}

log "Starting monitor for: ${NAMES[*]}"

while true; do
  for name in "${NAMES[@]}"; do
    jid="$(get_jobid_by_name "$name" || true)"
    state_file="$STATE_DIR/${name}.jid"

    if [[ -n "$jid" ]]; then
      echo -n "$jid" > "$state_file"
      log "$name running: JobID=$jid"
      continue
    fi

    # Not in squeue; check last known jobid
    if [[ -f "$state_file" ]]; then
      jid_saved="$(cat "$state_file")"
      if [[ -n "$jid_saved" ]]; then
        st="$(get_state_by_jobid "$jid_saved" || true)"
        if [[ -n "$st" ]]; then
          case "$st" in
            COMPLETED)
              log "$name completed (JobID=$jid_saved)" ;;
            FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
              log "$name failed with state=$st (JobID=$jid_saved). Inspecting logs..."
              tail_log_for_jobid "$jid_saved"
              log "Attempting resubmit for $name..."
              resubmit_by_name "$name" || true
              ;;
            *)
              # other terminal states or unknown
              log "$name state=$st (JobID=$jid_saved)" ;;
          esac
        else
          log "$name: no sacct record for JobID=$jid_saved yet"
        fi
      fi
    else
      log "$name: no state file yet; waiting for submission or start"
    fi
  done
  sleep 60
done



