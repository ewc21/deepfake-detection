#!/usr/bin/env bash
# Copy the SMALL result files (json/csv/png/yaml/log/md) from the group bucket into the repo's
# outputs/ folder so they can be committed. Checkpoints (*.pt) stay in the bucket.
#
#   bash scripts/tpu/collect_results.sh                 # from $DFF_OUTPUT_ROOT (default /mnt/group/outputs)
#   bash scripts/tpu/collect_results.sh /path/to/outputs
#
# Then: git add outputs && git commit -m "Phase 2 results" && git push
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${1:-${DFF_OUTPUT_ROOT:-/mnt/group/outputs}}"
DST="$PROJECT_DIR/outputs"
[ -d "$SRC" ] || { echo "No outputs at $SRC"; exit 1; }

python3 - "$SRC" "$DST" <<'EOF'
import shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
keep = {".json", ".csv", ".png", ".yaml", ".log", ".md"}
skip_dirs = {"checkpoints", "cache", "logs"}   # logs/ = raw console logs (train.log is kept per run)
n = size = 0
for p in src.rglob("*"):
    rel = p.relative_to(src)
    if not p.is_file() or p.suffix not in keep or skip_dirs & set(rel.parts):
        continue
    if rel.parts[0].startswith("smoke"):   # smoke tests are setup checks, not results
        continue
    out = dst / rel
    if out.exists() and out.stat().st_size == p.stat().st_size:
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, out)
    n += 1
    size += p.stat().st_size
print(f"Copied {n} new/changed files ({size / 1e6:.1f} MB) into {dst}")
EOF
echo "Checkpoints stay in $SRC (*/checkpoints/best.pt). Next: git add outputs && git commit && git push"
