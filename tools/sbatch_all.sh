#!/usr/bin/env bash
set -euo pipefail

# Usage: sbatch_all.sh [directory]
# Recursively submit all .sh scripts under the given directory via sbatch.
# If directory is omitted, you will be prompted to input one interactively.

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "Usage: $(basename "$0") <directory>"
  echo "Submit all .sh scripts under <directory> recursively with sbatch."
  exit 0
fi

TARGET_DIR=${1:-}
if [[ -z "$TARGET_DIR" ]]; then
  read -rp "Enter directory to scan for .sh scripts: " TARGET_DIR
fi

if [[ -z "$TARGET_DIR" ]]; then
  echo "Error: directory path is required." >&2
  exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: '$TARGET_DIR' is not a directory or does not exist." >&2
  exit 1
fi

readarray -t scripts < <(find "$TARGET_DIR" -type f -name "*.sh" | sort)

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "No .sh scripts found under: $TARGET_DIR"
  exit 0
fi

echo "Found ${#scripts[@]} scripts. Submitting with sbatch..."

for script_path in "${scripts[@]}"; do
  # Submit the script and capture output
  submit_out=""
  if ! submit_out=$(sbatch "$script_path" 2>&1); then
    echo "[FAIL] $script_path -> $submit_out" >&2
    continue
  fi

  # Typical output: "Submitted batch job 123456"
  job_id=$(sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' <<<"$submit_out")
  if [[ -n "$job_id" ]]; then
    echo "[OK] job=$job_id file=$script_path"
  else
    echo "[OK] file=$script_path -> $submit_out"
  fi
done

echo "All submissions attempted."


