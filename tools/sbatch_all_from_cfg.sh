#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /home/hpc/v100dd/v100dd12/code/3D-Mem/tools/sbatch_all_from_cfg.sh /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/exp_at
# Or pass a script root to submit all .sh directly:
#   bash /home/hpc/v100dd/v100dd12/code/3D-Mem/tools/sbatch_all_from_cfg.sh /home/hpc/v100dd/v100dd12/code/3D-Mem/script/exp_at

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cfg_root_or_script_root>" >&2
  exit 1
fi

INPUT_PATH="$1"
if [[ ! -d "$INPUT_PATH" ]]; then
  echo "Error: directory not found: $INPUT_PATH" >&2
  exit 1
fi

# If user passes a cfg root, map to the corresponding script root
if [[ "$INPUT_PATH" == *"/cfg/"* || "$INPUT_PATH" == *"/cfg" ]]; then
  # project root = everything before the first '/cfg/' segment
  PRJ_PREFIX="${INPUT_PATH%%/cfg/*}"
  # if INPUT_PATH ends exactly with '/cfg', REL should be empty
  if [[ "$INPUT_PATH" == */cfg ]]; then
    REL=""
  else
    REL="${INPUT_PATH#${PRJ_PREFIX}/cfg/}"
  fi
  SCRIPT_ROOT="${PRJ_PREFIX}/script/${REL}"
  CFG_ROOT="$INPUT_PATH"

  if [[ ! -d "$SCRIPT_ROOT" ]]; then
    echo "Error: mapped script root not found: $SCRIPT_ROOT" >&2
    exit 1
  fi

  echo "[INFO] Submitting jobs for all YAMLs under: $CFG_ROOT"
  echo "[INFO] Mapped script root: $SCRIPT_ROOT"

  # Traverse all YAML cfgs and submit corresponding sh
  while IFS= read -r -d '' yaml; do
    rel="${yaml#${CFG_ROOT}/}"
    base_no_ext="${rel%.yaml}"
    sh_path="${SCRIPT_ROOT}/${base_no_ext}.sh"
    if [[ -f "$sh_path" ]]; then
      echo "[SUBMIT] sbatch $sh_path"
      sbatch "$sh_path"
    else
      echo "[WARN] Missing sh for cfg: $yaml -> $sh_path" >&2
    fi
  done < <(find "$CFG_ROOT" -type f -name "*.yaml" -print0 | sort -z)

  exit 0
fi

# If user passes a script root, submit all .sh directly
if [[ "$INPUT_PATH" == *"/script/"* || "$INPUT_PATH" == *"/script" ]]; then
  SCRIPT_ROOT="$INPUT_PATH"
  echo "[INFO] Submitting all .sh under: $SCRIPT_ROOT"
  while IFS= read -r -d '' shf; do
    echo "[SUBMIT] sbatch $shf"
    sbatch "$shf"
  done < <(find "$SCRIPT_ROOT" -type f -name "*.sh" -print0 | sort -z)
  exit 0
fi

echo "Error: Input path must contain '/cfg' or '/script': $INPUT_PATH" >&2
exit 1


