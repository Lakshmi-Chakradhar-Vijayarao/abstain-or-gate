#!/usr/bin/env bash
# Build (and optionally push) the private Kaggle dataset + kernel for the Project 2 run.
#   KAGGLE_USER=<name> GOLDEN=<frozen split dir> ./scripts/kaggle/bundle.sh [--push]
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USER:?set KAGGLE_USER}"
GATEDRAG=${GATEDRAG:-../gated-llm-service}
GOLDEN=${GOLDEN:?set GOLDEN to a frozen split directory (dev.jsonl, test.jsonl, MANIFEST.json)}
D=dist/kaggle_p2/dataset; K=dist/kaggle_p2/kernel
rm -rf dist/kaggle_p2 && mkdir -p "$D/gatedrag/data/processed" "$D/abstain-or-gate" "$D/golden" "$K"
cp -R "$GATEDRAG/src" "$GATEDRAG/configs" "$GATEDRAG/pyproject.toml" "$D/gatedrag/"
rm -f "$D/gatedrag/configs/keys.toml"
cp "$GATEDRAG/data/processed/chunks.jsonl" "$D/gatedrag/data/processed/"
cp -R src pyproject.toml "$D/abstain-or-gate/"
cp "$GOLDEN"/dev.jsonl "$GOLDEN"/test.jsonl "$GOLDEN"/MANIFEST.json "$D/golden/"
find "$D" -name __pycache__ -prune -exec rm -rf {} +
cat > "$D/dataset-metadata.json" <<JSON
{"title": "abstain-or-gate-bundle", "id": "$KAGGLE_USER/abstain-or-gate-bundle", "licenses": [{"name": "Apache 2.0"}]}
JSON
cp scripts/kaggle/run_project2.py "$K/"
cat > "$K/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USER/abstain-or-gate-run", "title": "abstain-or-gate-run", "code_file": "run_project2.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true,
 "enable_internet": true, "dataset_sources": ["$KAGGLE_USER/abstain-or-gate-bundle"]}
JSON
echo "built dist/kaggle_p2 ($(du -sh "$D" | cut -f1))"
if [[ "${1:-}" == "--push" ]]; then
  if kaggle datasets status "$KAGGLE_USER/abstain-or-gate-bundle" >/dev/null 2>&1; then
    kaggle datasets version -p "$D" -m "bundle $(date +%F)" --dir-mode zip
  else
    kaggle datasets create -p "$D" --dir-mode zip
  fi
  echo "waiting for the dataset to finish processing..."; sleep 60
  kaggle kernels push -p "$K" --accelerator NvidiaTeslaT4
fi
