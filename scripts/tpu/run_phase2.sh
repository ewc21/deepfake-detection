#!/usr/bin/env bash
# Run all Phase 2 baselines on the TPU, resumably, then build the comparison.
#
#   source ~/dff_env.sh && tmux new -s train
#   bash scripts/tpu/run_phase2.sh --smoke                 # quick end-to-end check first
#   bash scripts/tpu/run_phase2.sh                         # 3 tasks x {plain, harmonize} x 3 seeds
#   bash scripts/tpu/run_phase2.sh --variants plain        # split work across 2 sessions:
#   bash scripts/tpu/run_phase2.sh --variants harmonize    #   one variant per machine
#
# Options:
#   --tasks "faceswap text2img combined"   tasks to run (default: all three)
#   --variants "plain harmonize"           which variants (default: both)
#   --seeds "0 1 2"                        seeds (default: 0 1 2)
#   --device tpu                           tpu (default) | cuda | mps | cpu | auto
#   --smoke                                1 epoch, 512 images/split, faceswap seed 0 -> experiment smoke_tpu
#   --dry-run                              print what would run, run nothing
#   anything after --  is passed to src.train, e.g.  -- --model resnet18 --set train.lr=3e-4
#
# Resumable: each (experiment, seed) runs as its own job and is SKIPPED if a finished run
# (metrics.json) already exists in $DFF_OUTPUT_ROOT. If a session expires mid-run, start a new
# session, run setup.sh, and re-run the same command: it continues with the missing seeds.
# Each job's console output is saved to $DFF_OUTPUT_ROOT/logs/.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
TASKS="faceswap text2img combined"
VARIANTS="plain harmonize"
SEEDS="0 1 2"
DEVICE="tpu"
SMOKE=0
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --tasks) TASKS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

: "${DFF_OUTPUT_ROOT:?Run 'source ~/dff_env.sh' first (sets DFF_OUTPUT_ROOT, DFF_ROOT, venv).}"
: "${DFF_ROOT:?Run 'source ~/dff_env.sh' first.}"
OUT="$DFF_OUTPUT_ROOT"
mkdir -p "$OUT/logs"

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "$DRY" = 0 ]; then
  echo -e "\033[33mWARNING: not inside tmux/screen — training stops if your SSH connection drops.\033[0m"
  echo "Recommended: tmux new -s train   (reattach later with: tmux attach -t train)"
  sleep 5
fi

if [ "$SMOKE" = 1 ]; then
  TASKS="faceswap"; VARIANTS="plain"; SEEDS="0"
  EXTRA+=(--experiment smoke_tpu --set train.epochs=1 data.subset_per_split=512)
fi

finished() {  # finished <experiment> <seed>: true if a run with metrics.json exists
  compgen -G "$OUT/$1/*_s$2_*/metrics.json" > /dev/null
}

experiment_name() {  # mirrors src.config.finalize_experiment_name
  if [ "$SMOKE" = 1 ]; then echo "smoke_tpu"; return; fi
  if [ "$2" = "harmonize" ]; then echo "baseline_$1_harmonized"; else echo "baseline_$1"; fi
}

declare -a DONE_EXPS=()
FAILED=0
START=$(date +%s)
for task in $TASKS; do
  for variant in $VARIANTS; do
    exp="$(experiment_name "$task" "$variant")"
    flags=(--task "$task" --device "$DEVICE")
    [ "$variant" = "harmonize" ] && flags+=(--harmonize)
    for seed in $SEEDS; do
      if [ "$SMOKE" = 0 ] && finished "$exp" "$seed"; then
        echo "[skip] $exp seed $seed (finished run exists)"
        continue
      fi
      log="$OUT/logs/${exp}_s${seed}_$(date +%Y%m%d-%H%M%S).log"
      cmd=(python -m src.train "${flags[@]}" --seeds "$seed" "${EXTRA[@]}")
      echo -e "\n\033[1m[run] $exp seed $seed\033[0m  ($(date +%H:%M))  log: $log"
      echo "      ${cmd[*]}"
      [ "$DRY" = 1 ] && continue
      "${cmd[@]}" 2>&1 | tee "$log"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ]; then
        echo -e "\033[31m[fail] $exp seed $seed — see $log\033[0m"
        FAILED=$((FAILED + 1))
        [ "$SMOKE" = 1 ] && exit 1
      fi
    done
    DONE_EXPS+=("$exp")
  done
done

[ "$DRY" = 1 ] && exit 0

echo -e "\n\033[1m==> Aggregating\033[0m"
for exp in "${DONE_EXPS[@]}"; do
  [ -d "$OUT/$exp" ] && python -m src.aggregate --experiment "$exp" --output-root "$OUT" || true
done

if [ "$SMOKE" = 1 ]; then
  echo -e "\n\033[1mSmoke test passed.\033[0m Epoch times (512 images/split) are in the log above."
  echo "Full data has 42,002 train + 8,998 val images per task, so scale epoch time by ~80x as a rough estimate."
else
  python -m src.compare --output-root "$OUT" || true
fi
MIN=$(( ($(date +%s) - START) / 60 ))
echo -e "\nDone in ${MIN} min. Failed jobs: $FAILED. Results in $OUT"
echo "Copy small result files into the repo: bash scripts/tpu/collect_results.sh"
[ "$FAILED" -eq 0 ]
