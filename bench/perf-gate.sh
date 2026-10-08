#!/usr/bin/env bash
# Performance gate: the serving benchmark (bench/rust) with this tree's
# fastokens against BASE_REF's, e.g. a pull request's source branch against its
# target branch. Exits non-zero on a regression.
#
#   bench/perf-gate.sh BASE_REF [gate options...]
#   bench/perf-gate.sh origin/main --models GLM-5.3 --rounds 3
#
# Both sides run *this* tree's harness: it is copied into a worktree of BASE_REF
# and built against that tree's fastokens, so only the library differs. If it
# does not compile there (an API the harness uses changed), BASE_REF's own
# harness is used instead. `gate` then alternates the two builds' workers; its
# options and verdict rules are in bench/rust/src/gate.rs.
set -euo pipefail

base_ref=${1:?usage: bench/perf-gate.sh BASE_REF [gate options...]}
shift
root=$(git rev-parse --show-toplevel)
base_sha=$(git -C "$root" rev-parse --verify "$base_ref^{commit}")
work=$root/target/perf-gate
base=$work/base
# One target dir for both builds, so the dependencies compile once.
export CARGO_TARGET_DIR=$root/bench/rust/target

git -C "$root" worktree remove --force "$base" 2>/dev/null || rm -rf "$base"
git -C "$root" worktree prune
mkdir -p "$work"
git -C "$root" worktree add --detach --quiet "$base" "$base_sha"
trap 'git -C "$root" worktree remove --force "$base"' EXIT

# How the report names the two sides (overridable, e.g. by CI).
head_label=${PERF_GATE_HEAD_LABEL:-$(git -C "$root" rev-parse --short HEAD)}
if [ -z "${PERF_GATE_HEAD_LABEL:-}" ] && [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
    head_label+="+dirty"
fi
base_label=${PERF_GATE_BASE_LABEL:-"$base_ref (${base_sha:0:7})"}

echo "--- building source: $head_label"
cargo build --release --manifest-path "$root/bench/rust/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/fastokens-serving-bench" "$work/source-bench"

echo "--- building target: $base_label"
rm -rf "$base/bench/rust"
mkdir -p "$base/bench/rust"
cp -R "$root/bench/rust/Cargo.toml" "$root/bench/rust/Cargo.lock" "$root/bench/rust/src" "$base/bench/rust/"
if ! cargo build --release --manifest-path "$base/bench/rust/Cargo.toml"; then
    echo "::warning title=Perf gate::this tree's bench/rust does not build against $base_ref; using $base_ref's own harness"
    rm -rf "$base/bench/rust"
    git -C "$base" checkout --quiet -- bench/rust
    cargo build --release --manifest-path "$base/bench/rust/Cargo.toml"
fi
cp "$CARGO_TARGET_DIR/release/fastokens-serving-bench" "$work/target-bench"

summary=()
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    summary=(--summary "$GITHUB_STEP_SUMMARY")
fi
"$work/source-bench" gate --base-worker "$work/target-bench" \
    --base-label "$base_label" --head-label "$head_label" \
    ${summary[@]+"${summary[@]}"} "$@"
