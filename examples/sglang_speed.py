#!/usr/bin/env python3
"""
Benchmark fastokens vs stock tokenizer in SGLang serving.

Launches SGLang servers with and without fastokens patching in parallel,
sends requests with max_tokens=1 using the ShareGPT dataset, and compares
prefill latency and throughput.

Usage:
    python benchmarks/sglang_bench.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    python benchmarks/sglang_bench.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --num-prompts 200 -- --tp 8
"""

from __future__ import annotations

import argparse
import atexit
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


def _wait_healthy(base_url: str, timeout: int, log_path: str) -> bool:
    """Poll /health until the server is ready or timeout."""
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)

    print("\nERROR: Server did not become healthy within timeout.", file=sys.stderr)
    try:
        with open(log_path) as f:
            lines = f.readlines()
        tail = lines[-80:]
        print("--- last 80 lines of server log ---", file=sys.stderr)
        for line in tail:
            print(line, end="", file=sys.stderr)
        print("--- end of server log ---", file=sys.stderr)
    except OSError:
        pass
    return False


def _launch_server(
    model: str,
    port: int,
    *,
    patched: bool = False,
    extra_args: list[str] | None = None,
    log_path: str,
) -> subprocess.Popen:
    if patched:
        cmd = [
            sys.executable,
            "-c",
            "import fastokens; fastokens.patch_transformers(); "
            "import runpy; runpy.run_module('sglang.launch_server', run_name='__main__')",
        ]
    else:
        cmd = [sys.executable, "-m", "sglang.launch_server"]

    cmd.extend(["--model-path", model, "--host", "0.0.0.0", "--port", str(port)])
    if extra_args:
        cmd.extend(extra_args)

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _active_procs.append(proc)
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
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
    base_url: str, model: str, prompt: str, endpoint: str,
) -> dict[str, float]:
    """Send one request with max_tokens=1 to the chosen endpoint."""
    if endpoint == "chat":
        url = f"{base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
    else:
        url = f"{base_url}/v1/completions"
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
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

    usage = resp_body.get("usage", {})
    return {
        "latency_ms": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "num_prompts": 1,
    }


def _send_batch(
    base_url: str, model: str, prompts: list[str],
) -> dict[str, float]:
    """Send a batch of prompts in one /v1/completions request."""
    url = f"{base_url}/v1/completions"
    body = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 1,
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

    usage = resp_body.get("usage", {})
    return {
        "latency_ms": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "num_prompts": len(prompts),
    }


def _run_bench(
    base_url: str,
    model: str,
    prompts: list[str],
    endpoint: str = "chat",
    batch_size: int = 1,
) -> dict[str, float]:
    """Send all prompts sequentially with max_tokens=1, return aggregate metrics."""
    results: list[dict[str, float]] = []
    errors = 0

    t_start = time.perf_counter()
    if batch_size > 1:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            try:
                results.append(_send_batch(base_url, model, batch))
            except Exception as exc:
                errors += 1
                print(f"  request failed: {exc}", file=sys.stderr)
    else:
        for prompt in prompts:
            try:
                results.append(_send_one(base_url, model, prompt, endpoint))
            except Exception as exc:
                errors += 1
                print(f"  request failed: {exc}", file=sys.stderr)
    duration_s = time.perf_counter() - t_start

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

    return metrics


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
) -> None:
    print()
    print("=" * _W)
    print("  SGLang Benchmark: baseline vs fastokens  (max_tokens=1)")
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
            "Benchmark fastokens vs stock tokenizer in SGLang serving. "
            "Sends /v1/chat/completions requests with max_tokens=1 to isolate "
            "prefill/tokenization overhead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s meta-llama/Llama-3.1-8B-Instruct
              %(prog)s meta-llama/Llama-3.1-8B-Instruct --num-prompts 100
              %(prog)s deepseek-ai/DeepSeek-V3 -- --tp 8
        """),
    )
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument(
        "--port", type=int, default=30000,
        help="Base port; baseline uses PORT, patched uses PORT+1 (default: 30000)",
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
        help="Prompts per request; >1 sends prompt arrays via /v1/completions "
             "(default: 1)",
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
        "--output", type=str, default=None,
        help="Save results as JSON to this path",
    )

    args = parser.parse_args(our_argv)

    if args.batch_size > 1 and args.endpoint == "chat":
        parser.error("--batch-size > 1 requires --endpoint completions")

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
    warmup_prompts = all_prompts[: args.warmup]
    bench_prompts = all_prompts[args.warmup :]
    print(f"  {len(bench_prompts)} benchmark + {len(warmup_prompts)} warmup prompts")

    ep_path = (
        "/v1/chat/completions" if args.endpoint == "chat"
        else "/v1/completions"
    )
    batch_info = f", batch_size={args.batch_size}" if args.batch_size > 1 else ""

    def _run_one(*, patched: bool, port: int) -> dict[str, float]:
        tag = "FASTOKENS" if patched else "BASELINE"
        base_url = f"http://127.0.0.1:{port}"

        print(f"\n  [{tag}] Launching SGLang server on port {port}...")

        log_fd, log_path = tempfile.mkstemp(
            prefix=f"sglang_{tag.lower()}_", suffix=".log",
            dir=tempfile.gettempdir(),
        )
        os.close(log_fd)

        proc = _launch_server(
            args.model, port, patched=patched,
            extra_args=server_extra or None, log_path=log_path,
        )

        try:
            print(f"  [{tag}] Waiting for server (log: {log_path})...")
            if not _wait_healthy(base_url, args.timeout, log_path):
                _stop_server(proc)
                sys.exit(1)

            if warmup_prompts:
                print(f"  [{tag}] Warming up ({len(warmup_prompts)} requests)...")
                _run_bench(
                    base_url, args.model, warmup_prompts,
                    args.endpoint, args.batch_size,
                )

            print(
                f"  [{tag}] Benchmarking ({len(bench_prompts)} prompts, "
                f"max_tokens=1, {ep_path}{batch_info})..."
            )
            metrics = _run_bench(
                base_url, args.model, bench_prompts,
                args.endpoint, args.batch_size,
            )
            _print_run_summary(tag, metrics)
            return metrics
        finally:
            print(f"  [{tag}] Stopping server...")
            _stop_server(proc)

    baseline_metrics: dict[str, float] | None = None
    patched_metrics: dict[str, float] | None = None

    if not args.patched_only:
        baseline_metrics = _run_one(patched=False, port=baseline_port)

    if not args.baseline_only:
        patched_metrics = _run_one(patched=True, port=patched_port)

    if baseline_metrics and patched_metrics:
        _print_comparison(args.model, baseline_metrics, patched_metrics)

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
