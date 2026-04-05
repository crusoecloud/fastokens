#!/usr/bin/env bash
# Validate a new model by running simple_bench against both datasets.
#
# Usage:
#   ./examples/validate_model.sh openai/gpt-oss-120b
#   ./examples/validate_model.sh Qwen/Qwen3-0.6B -n 50

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [extra args...]"
    echo "Example: $0 openai/gpt-oss-120b"
    echo "         $0 Qwen/Qwen3-0.6B -n 50"
    exit 1
fi

MODEL="$1"
shift
EXTRA_ARGS=("$@")

DATASETS=("RyokoAI/ShareGPT52K" "zai-org/LongBench-v2")

echo "=== Validating model: $MODEL ==="
echo ""

FAILED=0
for DATASET in "${DATASETS[@]}"; do
    echo "--- $DATASET ---"
    if cargo run --release --example simple_bench -- "$MODEL" --dataset "$DATASET" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
        echo "PASS: $MODEL on $DATASET"
    else
        echo "FAIL: $MODEL on $DATASET"
        FAILED=1
    fi
    echo ""
done

if [ "$FAILED" -eq 0 ]; then
    echo "=== All datasets passed for $MODEL ==="
else
    echo "=== FAILURES detected for $MODEL ==="
    exit 1
fi
