#!/usr/bin/env python3
"""
Benchmark fastokens vs stock tokenizer via Dynamo (frontend + SGLang backend).

Launches a Dynamo frontend and SGLang backend with and without fastokens env
vars, sends requests with max_tokens=1 using the ShareGPT dataset, and compares
prefill latency and throughput.

Usage:
    python examples/dynamo_speed.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    python examples/dynamo_speed.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --num-prompts 200 -- --enable-piecewise-cuda-graph --tp 2
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import statistics
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PORT = 0
_LAUNCHED_INSTANCES_COUNT = 1

_active_procs: list[subprocess.Popen] = []


def _cleanup() -> None:
    for proc in _active_procs:
        _kill(proc)


atexit.register(_cleanup)


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _wait_healthy(base_url: str, timeout: int, log_paths: list[str]) -> bool:
    """Poll /health until the server is ready or timeout."""
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = json.loads(resp.read())
                if len(body.get("instances")) == _LAUNCHED_INSTANCES_COUNT:
                    return True
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            pass
        time.sleep(3)

    print("\nERROR: Server did not become healthy within timeout.", file=sys.stderr)
    for log_path in log_paths:
        try:
            with open(log_path) as f:
                lines = f.readlines()
            tail = lines[-80:]
            print(f"--- last 80 lines of {log_path} ---", file=sys.stderr)
            for line in tail:
                print(line, end="", file=sys.stderr)
            print("--- end of log ---", file=sys.stderr)
        except OSError:
            pass
    return False


def _launch_dynamo(
    model: str,
    port: int,
    *,
    patched: bool = False,
    extra_args: list[str] | None = None,
    frontend_log_path: str,
    backend_log_path: str,
) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Launch a Dynamo frontend + SGLang backend pair."""
    # --- Backend (SGLang via dynamo) ---
    backend_cmd = [
        sys.executable, "-m", "dynamo.sglang",
        "--model-path", model,
        "--discovery-backend", "file",
    ]
    if extra_args:
        backend_cmd.extend(extra_args)

    backend_env = os.environ.copy()
    backend_env["DYN_SYSTEM_PORT"] = str(_DEFAULT_SYSTEM_PORT)
    backend_env["DYN_SGL_USE_TOKENIZER"] = "1"
    if patched:
        backend_env["DYN_USE_FASTOKENS"] = "1"

    backend_log = open(backend_log_path, "w")
    backend_proc = subprocess.Popen(
        backend_cmd,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        env=backend_env,
    )
    _active_procs.append(backend_proc)

    # --- Frontend (Dynamo HTTP router) ---
    frontend_cmd = [
        sys.executable, "-m", "dynamo.frontend",
        "--http-port", str(port),
        "--discovery-backend", "file",
    ]

    frontend_env = os.environ.copy()
    frontend_env["DYN_SYSTEM_STARTING_HEALTH_STATUS"] = "notready"
    frontend_env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'

    frontend_log = open(frontend_log_path, "w")
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
        env=frontend_env,
    )
    _active_procs.append(frontend_proc)

    return frontend_proc, backend_proc


def _stop_dynamo(
    frontend: subprocess.Popen, backend: subprocess.Popen,
) -> None:
    for proc in (frontend, backend):
        _kill(proc)
        if proc in _active_procs:
            _active_procs.remove(proc)
    time.sleep(3)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_DATASET_URLS = {
    "sharegpt": (
        "https://huggingface.co/datasets/anon8231489123/"
        "ShareGPT_Vicuna_unfiltered/resolve/main/"
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
    ),
    "longbench": (
        "https://huggingface.co/datasets/zai-org/"
        "LongBench-v2/resolve/main/data.json",
        "LongBench-v2_data.json",
    ),
}


def _download_dataset(name: str) -> list[dict]:
    """Download a dataset, caching locally."""
    url, filename = _DATASET_URLS[name]
    cache = os.path.join(tempfile.gettempdir(), "fastokens_bench_cache")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, filename)
    if not os.path.exists(path):
        print(f"  Downloading {name} dataset...")
        urllib.request.urlretrieve(url, path)
    with open(path) as f:
        return json.load(f)


def _extract_prompt_sharegpt(item: dict) -> str | None:
    convs = item.get("conversations", [])
    if not convs:
        return None
    first = convs[0]
    if first.get("from") != "human":
        return None
    text = first.get("value", "").strip()
    return text or None


def _extract_prompt_longbench(item: dict) -> str | None:
    context = item.get("context", "").strip()
    return context or None


_EXTRACTORS = {
    "sharegpt": _extract_prompt_sharegpt,
    "longbench": _extract_prompt_longbench,
}


def _sample_prompts(
    dataset: list[dict], num_prompts: int, min_len: int = 0,
    dataset_name: str = "sharegpt",
) -> list[str]:
    """Extract text prompts from the dataset."""
    extract = _EXTRACTORS[dataset_name]
    prompts: list[str] = []
    for item in dataset:
        if len(prompts) >= num_prompts:
            break
        text = extract(item)
        if not text or len(text) < min_len:
            continue
        prompts.append(text)
    return prompts


# ---------------------------------------------------------------------------
# Token-length bucketing
# ---------------------------------------------------------------------------

_BUCKET_BOUNDS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
_BUCKET_LABELS = ["0-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-32k", "32-64k", "64-128k"]


def _format_tokens(n: int) -> str:
    """Format token count for display: 512, 1k, 2k, 16k, etc."""
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def _parse_input_distribution(spec_str: str) -> dict[int, int]:
    """Parse a percentile distribution string like '{10: 512, 50: 2048, 100: 8192}'.

    Returns dict mapping percentile -> token count, sorted by percentile.
    """
    import ast
    parsed = ast.literal_eval(spec_str)
    if not isinstance(parsed, dict):
        raise ValueError(
            "--input-distribution must be a dict, "
            "e.g. '{50: 1024, 90: 4096, 100: 16384}'"
        )
    result: dict[int, int] = {}
    for k, v in parsed.items():
        pct, tokens = int(k), int(v)
        if not 0 <= pct <= 100:
            raise ValueError(f"Percentile {pct} must be in [0, 100]")
        if tokens <= 0:
            raise ValueError(f"Token count must be positive, got {tokens}")
        result[pct] = tokens
    if 100 not in result:
        raise ValueError("Distribution must include percentile 100")
    # Validate monotonically increasing
    pcts = sorted(result.keys())
    vals = [result[p] for p in pcts]
    if vals != sorted(vals):
        raise ValueError("Token counts must be non-decreasing with percentile")
    return dict((p, result[p]) for p in pcts)


def _adjust_tokens(token_ids: list[int], target: int) -> list[int]:
    """Truncate or repeat *token_ids* to reach exactly *target* length."""
    if len(token_ids) >= target:
        return token_ids[:target]
    reps = (target // len(token_ids)) + 1
    return (token_ids * reps)[:target]


def _prepare_bucketed_prompts(
    model: str, prompts: list[str],
    shared_prefix: float = 0.0,
) -> tuple[list[tuple[str, str]], list[int], list[str]]:
    """Create a version of each prompt for every bucket by truncating or repeating.

    If *shared_prefix* > 0, the first fraction of tokens is decoded as a
    separate system-prompt string so the server can cache it independently.

    Returns (adjusted_prompts, bucket_indices, bucket_labels).
    Each prompt is a (system_text, user_text) tuple.
    """
    from transformers import AutoTokenizer
    print(f"  Preparing bucketed prompts from {len(prompts)} source prompts...")
    if shared_prefix > 0:
        print(f"    shared prefix: {shared_prefix:.0%}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    all_token_ids = [
        ids for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
        if ids
    ]

    bucket_lower = [0] + _BUCKET_BOUNDS[:-1]  # [0, 1024, 2048, ...]

    adjusted_prompts: list[tuple[str, str]] = []
    bucket_indices: list[int] = []

    for bucket_idx, (lo, hi) in enumerate(zip(bucket_lower, _BUCKET_BOUNDS)):
        target = (lo + hi) // 2 if lo > 0 else hi // 2
        prefix_len = int(target * shared_prefix)
        suffix_len = target - prefix_len

        prefix_text = ""
        if prefix_len > 0:
            prefix_ids = _adjust_tokens(all_token_ids[0], prefix_len)
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

        count = 0
        for token_ids in all_token_ids:
            if prefix_len > 0:
                suffix_ids = _adjust_tokens(token_ids, suffix_len)
            else:
                suffix_ids = _adjust_tokens(token_ids, target)
            suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            adjusted_prompts.append((prefix_text, suffix_text))
            bucket_indices.append(bucket_idx)
            count += 1
        print(f"    {_BUCKET_LABELS[bucket_idx]}: {count} prompts (target {target} tokens)")

    return adjusted_prompts, bucket_indices, list(_BUCKET_LABELS)


def _prepare_distributed_prompts(
    model: str,
    prompts: list[str],
    distribution: dict[int, int],
    num_prompts: int,
    shared_prefix: float = 0.0,
) -> tuple[list[tuple[str, str]], list[int], list[str]]:
    """Create prompts following a percentile-based token length distribution.

    *distribution* maps percentile -> token count, e.g. {50: 1024, 90: 4096, 100: 16384}.
    *num_prompts* total prompts are allocated across buckets proportionally.

    Returns (adjusted_prompts, bucket_indices, bucket_labels).
    Each prompt is a (system_text, user_text) tuple.
    """
    from transformers import AutoTokenizer

    percentiles = sorted(distribution.keys())
    bucket_labels: list[str] = []
    bucket_targets: list[int] = []
    bucket_counts: list[int] = []
    unrounded: list[float] = []

    prev_pct = 0
    for pct in percentiles:
        width = pct - prev_pct
        count = round(num_prompts * width / 100.0)
        tokens = distribution[pct]
        bucket_labels.append(f"p{pct}({_format_tokens(tokens)})")
        bucket_targets.append(tokens)
        bucket_counts.append(count)
        unrounded.append(num_prompts * width / 100.0)
        prev_pct = pct

    # Fix rounding with largest-remainder method
    total = sum(bucket_counts)
    remaining = num_prompts - total
    if remaining != 0:
        errors = [c - uc for c, uc in zip(bucket_counts, unrounded)]
        if remaining > 0:
            for i in sorted(range(len(bucket_counts)), key=lambda i: errors[i]):
                if remaining <= 0:
                    break
                bucket_counts[i] += 1
                remaining -= 1
        else:
            for i in sorted(range(len(bucket_counts)), key=lambda i: -errors[i]):
                if remaining >= 0:
                    break
                if bucket_counts[i] > 0:
                    bucket_counts[i] -= 1
                    remaining += 1

    print(f"  Preparing distributed prompts from {len(prompts)} source prompts...")
    if shared_prefix > 0:
        print(f"    shared prefix: {shared_prefix:.0%}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    all_token_ids = [
        ids for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
        if ids
    ]

    adjusted_prompts: list[tuple[str, str]] = []
    bucket_indices: list[int] = []

    for bucket_idx, (target, count) in enumerate(zip(bucket_targets, bucket_counts)):
        prefix_len = int(target * shared_prefix)
        suffix_len = target - prefix_len

        prefix_text = ""
        if prefix_len > 0:
            prefix_ids = _adjust_tokens(all_token_ids[0], prefix_len)
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)

        for j in range(count):
            token_ids = all_token_ids[j % len(all_token_ids)]
            if prefix_len > 0:
                suffix_ids = _adjust_tokens(token_ids, suffix_len)
            else:
                suffix_ids = _adjust_tokens(token_ids, target)
            suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            adjusted_prompts.append((prefix_text, suffix_text))
            bucket_indices.append(bucket_idx)
        print(f"    {bucket_labels[bucket_idx]}: {count} prompts (target {target} tokens)")

    return adjusted_prompts, bucket_indices, bucket_labels


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _send_one(
    base_url: str, model: str, prompt: tuple[str, str], endpoint: str,
    max_tokens: int = 1,
) -> dict[str, float]:
    """Send one request to the chosen endpoint.

    *prompt* is a (system_text, user_text) tuple.  When *system_text* is
    non-empty and the endpoint is ``chat``, it is sent as a system message
    so the server can cache the shared prefix independently.
    """
    system_text, user_text = prompt
    if endpoint == "chat":
        url = f"{base_url}/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0,
        }
    else:
        url = f"{base_url}/v1/completions"
        body = {
            "model": model,
            "prompt": system_text + user_text,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0,
        }

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp_body = json.loads(resp.read())
    latency = (time.perf_counter() - t0) * 1000

    usage = resp_body.get("usage") or {}
    return {
        "latency_ms": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "num_prompts": 1,
    }


_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


def _retry(fn, *args, **kwargs):
    """Call *fn* with retries and exponential backoff."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))


def _print_progress(done: int, total: int, errors: int, width: int = 40) -> None:
    filled = int(width * done / total) if total else width
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = done / total * 100 if total else 100
    err_str = f" ({errors} err)" if errors else ""
    print(f"\r  [{bar}] {done}/{total} {pct:5.1f}%{err_str}", end="", flush=True)


def _run_bench(
    base_url: str,
    model: str,
    prompts: list[tuple[str, str]],
    endpoint: str = "chat",
    batch_size: int = 1,
    max_tokens: int = 1,
) -> tuple[dict[str, float], list[float]]:
    """Send prompts up to *batch_size* requests concurrently.

    Returns aggregate metrics and per-prompt latencies (−1 for failures).
    """
    total = len(prompts)
    per_prompt_latencies: list[float] = [-1.0] * total
    results_by_idx: dict[int, dict[str, float]] = {}
    errors = 0
    done_count = 0

    def _do_one(idx: int) -> tuple[int, dict[str, float]]:
        return idx, _retry(_send_one, base_url, model, prompts[idx], endpoint, max_tokens)

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = {pool.submit(_do_one, i): i for i in range(total)}
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            try:
                idx, r = future.result()
                results_by_idx[idx] = r
                per_prompt_latencies[idx] = r["latency_ms"]
            except Exception as exc:
                errors += 1
                print(
                    f"\n  request failed after {_MAX_RETRIES} retries: {exc}",
                    file=sys.stderr,
                )
            _print_progress(done_count, total, errors)
    print()  # newline after progress bar
    duration_s = time.perf_counter() - t_start

    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    latencies = [r["latency_ms"] for r in results]
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_prompts = sum(int(r["num_prompts"]) for r in results)
    n = len(results)

    metrics: dict[str, float] = {
        "successful_requests": n,
        "successful_prompts": total_prompts,
        "failed_requests": errors,
        "duration_s": duration_s,
        "total_input_tokens": total_prompt_tokens,
    }
    if n > 0:
        metrics["request_throughput"] = n / duration_s
        metrics["prompt_throughput"] = total_prompts / duration_s
        metrics["input_throughput"] = total_prompt_tokens / duration_s
        metrics["mean_latency_ms"] = statistics.mean(latencies)
        metrics["median_latency_ms"] = statistics.median(latencies)
        metrics["p99_latency_ms"] = _percentile(latencies, 99)

    return metrics, per_prompt_latencies


# ---------------------------------------------------------------------------
# Tokenize-only benchmark
# ---------------------------------------------------------------------------


def _run_tokenize_bench(
    model: str,
    prompts: list[tuple[str, str]],
    *,
    use_fastokens: bool = False,
) -> tuple[dict[str, float], list[float]]:
    """Benchmark tokenization speed only (no server).

    Returns (metrics, per_prompt_latencies) in the same shape as _run_bench.
    """
    from transformers import AutoTokenizer

    if use_fastokens:
        import fastokens
        fastokens.patch_transformers()

    tokenizer = AutoTokenizer.from_pretrained(model)

    total = len(prompts)
    per_prompt_latencies: list[float] = []
    total_tokens = 0

    t_start = time.perf_counter()
    for idx, (sys_text, user_text) in enumerate(prompts):
        t0 = time.perf_counter()
        ids = tokenizer.encode(sys_text + user_text)
        latency = (time.perf_counter() - t0) * 1000
        per_prompt_latencies.append(latency)
        total_tokens += len(ids)
        _print_progress(idx + 1, total, 0)
    print()
    duration_s = time.perf_counter() - t_start

    n = len(per_prompt_latencies)
    metrics: dict[str, float] = {
        "successful_requests": n,
        "successful_prompts": n,
        "failed_requests": 0,
        "duration_s": duration_s,
        "total_input_tokens": total_tokens,
    }
    if n > 0:
        metrics["request_throughput"] = n / duration_s
        metrics["prompt_throughput"] = n / duration_s
        metrics["input_throughput"] = total_tokens / duration_s
        metrics["mean_latency_ms"] = statistics.mean(per_prompt_latencies)
        metrics["median_latency_ms"] = statistics.median(per_prompt_latencies)
        metrics["p99_latency_ms"] = _percentile(per_prompt_latencies, 99)

    return metrics, per_prompt_latencies


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

# (label, metric key, higher_is_better)
_TABLE_ROWS: list[tuple[str, str | None, bool | None]] = [
    ("Request throughput (req/s)", "request_throughput", True),
    ("Prompt throughput (prompts/s)", "prompt_throughput", True),
    ("Input tok throughput (tok/s)", "input_throughput", True),
    ("", None, None),
    ("Mean latency (ms)", "mean_latency_ms", False),
    ("Median latency (ms)", "median_latency_ms", False),
    ("P99 latency (ms)", "p99_latency_ms", False),
]

_W = 70


def _print_comparison(
    model: str,
    baseline: dict[str, float],
    patched: dict[str, float],
    title: str = "Dynamo Benchmark: baseline vs fastokens",
) -> None:
    print()
    print("=" * _W)
    print(f"  {title}")
    print(f"  Model: {model}")
    print("=" * _W)
    print(f"  {'Metric':<32} {'Baseline':>12} {'Fastokens':>12} {'Change':>10}")
    print("-" * _W)

    for label, key, higher_is_better in _TABLE_ROWS:
        if key is None:
            print("-" * _W)
            continue

        b = baseline.get(key)
        p = patched.get(key)
        if b is None or p is None:
            print(f"  {label:<32} {'N/A':>12} {'N/A':>12} {'':>10}")
            continue

        b_str = f"{b:,.2f}"
        p_str = f"{p:,.2f}"

        if b != 0:
            pct = ((p - b) / b * 100) if higher_is_better else ((b - p) / b * 100)
            change = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
        else:
            change = ""

        print(f"  {label:<32} {b_str:>12} {p_str:>12} {change:>10}")

    print("=" * _W)


def _print_bucket_comparison(
    model: str,
    baseline_latencies: list[float],
    patched_latencies: list[float],
    bucket_indices: list[int],
    bucket_labels: list[str],
) -> None:
    """Print P50 latency by token-length bucket for baseline vs patched."""
    from collections import defaultdict

    baseline_by_bucket: dict[int, list[float]] = defaultdict(list)
    patched_by_bucket: dict[int, list[float]] = defaultdict(list)

    for i, bucket in enumerate(bucket_indices):
        if i < len(baseline_latencies) and baseline_latencies[i] >= 0:
            baseline_by_bucket[bucket].append(baseline_latencies[i])
        if i < len(patched_latencies) and patched_latencies[i] >= 0:
            patched_by_bucket[bucket].append(patched_latencies[i])

    # Determine column width based on longest label
    label_w = max(12, max((len(l) for l in bucket_labels), default=12) + 2)
    row_w = max(_W, label_w + 48)

    print()
    print("=" * row_w)
    print("  P50 Latency by Token-Length Bucket")
    print(f"  Model: {model}")
    print("=" * row_w)
    print(f"  {'Bucket':<{label_w}} {'N':>6} {'Baseline P50':>14} {'Fastokens P50':>14} {'Change':>10}")
    print("-" * row_w)

    for i, label in enumerate(bucket_labels):
        b_lats = baseline_by_bucket.get(i, [])
        p_lats = patched_by_bucket.get(i, [])
        if not b_lats and not p_lats:
            continue

        n = max(len(b_lats), len(p_lats))
        b_p50 = statistics.median(b_lats) if b_lats else None
        p_p50 = statistics.median(p_lats) if p_lats else None

        b_str = f"{b_p50:,.1f} ms" if b_p50 is not None else "N/A"
        p_str = f"{p_p50:,.1f} ms" if p_p50 is not None else "N/A"

        if b_p50 and p_p50 and b_p50 != 0:
            pct = (b_p50 - p_p50) / b_p50 * 100  # lower is better
            change = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
        else:
            change = ""

        print(f"  {label:<{label_w}} {n:>6} {b_str:>14} {p_str:>14} {change:>10}")

    print("=" * row_w)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_at_double_dash(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at '--', returning (our_args, extra_server_args)."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _print_run_summary(tag: str, metrics: dict[str, float]) -> None:
    n = int(metrics.get("successful_requests", 0))
    fails = int(metrics.get("failed_requests", 0))
    dur = metrics.get("duration_s", 0)
    print(f"\n  [{tag}] {n} requests in {dur:.1f}s", end="")
    if fails:
        print(f" ({fails} failed)", end="")
    print()
    if "mean_latency_ms" in metrics:
        print(f"    mean latency:   {metrics['mean_latency_ms']:.1f} ms")
        print(f"    median latency: {metrics['median_latency_ms']:.1f} ms")
        print(f"    p99 latency:    {metrics['p99_latency_ms']:.1f} ms")
    if "request_throughput" in metrics:
        parts = [f"{metrics['request_throughput']:.2f} req/s"]
        if "prompt_throughput" in metrics and metrics["prompt_throughput"] != metrics["request_throughput"]:
            parts.append(f"{metrics['prompt_throughput']:.2f} prompts/s")
        parts.append(f"{metrics.get('input_throughput', 0):.0f} tok/s")
        print(f"    throughput:     {', '.join(parts)}")


def main(argv: list[str] | None = None) -> None:
    our_argv, server_extra = _split_at_double_dash(
        argv if argv is not None else sys.argv[1:]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fastokens vs stock tokenizer via Dynamo "
            "(frontend + SGLang backend). Sends requests with max_tokens=1 "
            "to isolate prefill/tokenization overhead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s meta-llama/Llama-3.1-8B-Instruct
              %(prog)s meta-llama/Llama-3.1-8B-Instruct --num-prompts 100
              %(prog)s deepseek-ai/DeepSeek-V3 -- --enable-piecewise-cuda-graph --tp 8
        """),
    )
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Dynamo frontend HTTP port; baseline uses PORT, patched uses PORT+1 "
             "(default: 8000)",
    )
    parser.add_argument(
        "--dataset", choices=["sharegpt", "longbench"], default="sharegpt",
        help="Dataset to use (default: sharegpt)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=-1,
        help="Number of prompts to benchmark (-1 = all, default: -1)",
    )
    parser.add_argument(
        "--endpoint", choices=["chat", "completions"], default="chat",
        help="API endpoint: 'chat' for /v1/chat/completions, "
             "'completions' for /v1/completions (default: chat)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of concurrent requests in flight (default: 1)",
    )
    parser.add_argument(
        "--output-len", type=int, default=1,
        help="Exact number of output tokens per request (default: 1)",
    )
    parser.add_argument(
        "--min-input-len", type=int, default=0,
        help="Drop prompts shorter than this many characters (default: 0)",
    )
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Warmup requests before measuring (default: 100)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Server startup timeout in seconds (default: 600)",
    )

    parser.add_argument(
        "--baseline-url", type=str, default=None,
        help="Use a pre-existing baseline server at this URL (e.g. http://host:30000) "
             "instead of launching one",
    )
    parser.add_argument(
        "--patched-url", type=str, default=None,
        help="Use a pre-existing patched server at this URL (e.g. http://host:30001) "
             "instead of launching one",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--baseline-only", action="store_true",
        help="Only run the baseline (unpatched) benchmark",
    )
    group.add_argument(
        "--patched-only", action="store_true",
        help="Only run the patched (fastokens) benchmark",
    )

    parser.add_argument(
        "--input-distribution", type=str, default=None,
        help="Percentile-based token distribution as a dict string, e.g. "
             "'{10: 512, 50: 2048, 90: 8192, 100: 32768}'. "
             "Overrides the default exponential buckets.",
    )
    parser.add_argument(
        "--input-len", type=int, default=None,
        help="Fixed input length in tokens. All prompts will be exactly this "
             "many tokens (single bucket). Overrides --input-distribution.",
    )
    parser.add_argument(
        "--shared-prefix", type=float, default=0.0,
        help="Fraction of input tokens shared across all prompts in a bucket "
             "(0.0-1.0). E.g. 0.9 means 90%% of tokens are identical prefix, "
             "10%% are unique suffix. Requires --input-len or --input-distribution.",
    )
    parser.add_argument(
        "--tokenize-only", action="store_true",
        help="Compare tokenization speed only (stock vs fastokens), "
             "without launching any servers",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results as JSON to this path",
    )

    args = parser.parse_args(our_argv)

    baseline_port = args.port
    patched_port = args.port + 1

    # Load dataset once, shared across both runs.
    print(f"Loading {args.dataset} dataset...")
    dataset = _download_dataset(args.dataset)
    if args.num_prompts < 0:
        all_prompts = _sample_prompts(
            dataset, len(dataset), args.min_input_len, args.dataset,
        )
    else:
        all_prompts = _sample_prompts(
            dataset, args.num_prompts + args.warmup, args.min_input_len,
            args.dataset,
        )
    warmup_prompts: list[tuple[str, str]] = [("", p) for p in all_prompts[: args.warmup]]
    source_prompts = all_prompts[args.warmup :]
    print(f"  {len(source_prompts)} source prompts + {len(warmup_prompts)} warmup prompts")

    # Create adjusted prompts for each bucket (truncated or repeated to fit).
    if args.input_len is not None:
        distribution = {100: args.input_len}
        bench_prompts, bucket_indices, bucket_labels = _prepare_distributed_prompts(
            args.model, source_prompts, distribution, len(source_prompts),
            shared_prefix=args.shared_prefix,
        )
    elif args.input_distribution:
        distribution = _parse_input_distribution(args.input_distribution)
        bench_prompts, bucket_indices, bucket_labels = _prepare_distributed_prompts(
            args.model, source_prompts, distribution, len(source_prompts),
            shared_prefix=args.shared_prefix,
        )
    else:
        bench_prompts, bucket_indices, bucket_labels = _prepare_bucketed_prompts(
            args.model, source_prompts,
            shared_prefix=args.shared_prefix,
        )

    # ---- Tokenize-only mode: compare tokenizers without servers ----
    if args.tokenize_only:
        baseline_metrics: dict[str, float] | None = None
        patched_metrics: dict[str, float] | None = None
        baseline_latencies: list[float] = []
        patched_latencies: list[float] = []

        if not args.patched_only:
            print(f"\n  [BASELINE] Tokenizing {len(bench_prompts)} prompts (stock tokenizer)...")
            baseline_metrics, baseline_latencies = _run_tokenize_bench(
                args.model, bench_prompts, use_fastokens=False,
            )
            _print_run_summary("BASELINE", baseline_metrics)

        if not args.baseline_only:
            print(f"\n  [FASTOKENS] Tokenizing {len(bench_prompts)} prompts (fastokens)...")
            patched_metrics, patched_latencies = _run_tokenize_bench(
                args.model, bench_prompts, use_fastokens=True,
            )
            _print_run_summary("FASTOKENS", patched_metrics)

        if baseline_metrics and patched_metrics:
            _print_comparison(
                args.model, baseline_metrics, patched_metrics,
                title="Tokenization Benchmark: stock vs fastokens",
            )
            _print_bucket_comparison(
                args.model, baseline_latencies, patched_latencies,
                bucket_indices, bucket_labels,
            )

        if args.output:
            results: dict = {"model": args.model, "mode": "tokenize-only"}
            if baseline_metrics:
                results["baseline"] = baseline_metrics
            if patched_metrics:
                results["fastokens"] = patched_metrics
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        return

    ep_path = (
        "/v1/chat/completions" if args.endpoint == "chat"
        else "/v1/completions"
    )
    conc_info = f", concurrency={args.batch_size}" if args.batch_size > 1 else ""

    def _run_one(*, patched: bool, port: int, external_url: str | None = None) -> tuple[dict[str, float], list[float]]:
        tag = "FASTOKENS" if patched else "BASELINE"

        if external_url is not None:
            base_url = external_url.rstrip("/")
            print(f"\n  [{tag}] Using external server at {base_url}")

            if warmup_prompts:
                print(f"  [{tag}] Warming up ({len(warmup_prompts)} requests)...")
                _run_bench(
                    base_url, args.model, warmup_prompts,
                    args.endpoint, args.batch_size, args.output_len,
                )

            print(
                f"  [{tag}] Benchmarking ({len(bench_prompts)} prompts, "
                f"max_tokens={args.output_len}, {ep_path}{conc_info})..."
            )
            metrics, latencies = _run_bench(
                base_url, args.model, bench_prompts,
                args.endpoint, args.batch_size, args.output_len,
            )
            _print_run_summary(tag, metrics)
            return metrics, latencies

        base_url = f"http://127.0.0.1:{port}"
        print(f"\n  [{tag}] Launching Dynamo frontend + backend on port {port}...")

        tmpdir = tempfile.gettempdir()
        prefix = f"dynamo_{tag.lower()}_"
        fe_fd, fe_log = tempfile.mkstemp(prefix=prefix + "fe_", suffix=".log", dir=tmpdir)
        os.close(fe_fd)
        be_fd, be_log = tempfile.mkstemp(prefix=prefix + "be_", suffix=".log", dir=tmpdir)
        os.close(be_fd)

        frontend_proc, backend_proc = _launch_dynamo(
            args.model, port, patched=patched,
            extra_args=server_extra or None,
            frontend_log_path=fe_log,
            backend_log_path=be_log,
        )

        try:
            print(f"  [{tag}] Waiting for server (frontend: {fe_log}, backend: {be_log})...")
            if not _wait_healthy(base_url, args.timeout, [fe_log, be_log]):
                _stop_dynamo(frontend_proc, backend_proc)
                sys.exit(1)

            if warmup_prompts:
                print(f"  [{tag}] Warming up ({len(warmup_prompts)} requests)...")
                _run_bench(
                    base_url, args.model, warmup_prompts,
                    args.endpoint, args.batch_size, args.output_len,
                )

            print(
                f"  [{tag}] Benchmarking ({len(bench_prompts)} prompts, "
                f"max_tokens={args.output_len}, {ep_path}{conc_info})..."
            )
            metrics, latencies = _run_bench(
                base_url, args.model, bench_prompts,
                args.endpoint, args.batch_size, args.output_len,
            )
            _print_run_summary(tag, metrics)
            return metrics, latencies
        finally:
            print(f"  [{tag}] Stopping Dynamo...")
            _stop_dynamo(frontend_proc, backend_proc)

    baseline_metrics: dict[str, float] | None = None
    patched_metrics: dict[str, float] | None = None
    baseline_latencies: list[float] = []
    patched_latencies: list[float] = []

    if not args.patched_only:
        baseline_metrics, baseline_latencies = _run_one(
            patched=False, port=baseline_port,
            external_url=args.baseline_url,
        )

    if not args.baseline_only:
        patched_metrics, patched_latencies = _run_one(
            patched=True, port=patched_port,
            external_url=args.patched_url,
        )

    if baseline_metrics and patched_metrics:
        _print_comparison(args.model, baseline_metrics, patched_metrics)
        _print_bucket_comparison(
            args.model, baseline_latencies, patched_latencies,
            bucket_indices, bucket_labels,
        )

    if args.output:
        results: dict = {"model": args.model, "num_prompts": args.num_prompts}
        if baseline_metrics:
            results["baseline"] = baseline_metrics
        if patched_metrics:
            results["fastokens"] = patched_metrics
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
