#!/usr/bin/env python3
"""
Compare eval quality between fastokens-patched and stock SGLang tokenizer.

Sequentially launches SGLang servers with and without fastokens patching,
runs the same evaluation benchmark on both, and compares scores.

Usage:
    python benchmarks/quality.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
    python benchmarks/quality.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --eval gpqa --num-examples 50
    python benchmarks/quality.py nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 -- --tp 8
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Process management (mirrors sglang_bench.py)
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
# Eval runners
# ---------------------------------------------------------------------------

_EVAL_DATASETS = {
    "gpqa": "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
    "mmlu": "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
}


def _build_eval(eval_name: str, num_examples: int | None, num_threads: int):
    """Build an sglang eval object by name."""
    if eval_name == "gpqa":
        from sglang.test.simple_eval_gpqa import GPQAEval

        return GPQAEval(
            _EVAL_DATASETS["gpqa"], num_examples, num_threads,
        )
    elif eval_name == "mmlu":
        from sglang.test.simple_eval_mmlu import MMLUEval

        return MMLUEval(
            _EVAL_DATASETS["mmlu"], num_examples, num_threads,
        )
    elif eval_name == "humaneval":
        from sglang.test.simple_eval_humaneval import HumanEval

        return HumanEval(num_examples, num_threads)
    elif eval_name == "gsm8k":
        from sglang.test.simple_eval_gsm8k import GSM8KEval

        return GSM8KEval(num_examples=num_examples, num_threads=num_threads)
    elif eval_name == "mgsm":
        from sglang.test.simple_eval_mgsm import MGSMEval

        return MGSMEval(num_examples, num_threads)
    else:
        raise ValueError(f"Unknown eval: {eval_name}")


def _run_eval(
    base_url: str,
    model: str,
    eval_name: str,
    num_examples: int | None,
    num_threads: int,
) -> dict:
    """Run an eval against a server and return {score, metrics, latency_s}."""
    from sglang.test.simple_eval_common import ChatCompletionSampler, set_ulimit

    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    sampler = ChatCompletionSampler(
        base_url=f"{base_url}/v1",
        model=model,
    )

    eval_obj = _build_eval(eval_name, num_examples, num_threads)

    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    return {
        "score": result.score,
        "metrics": result.metrics,
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_W = 70


def _print_comparison(
    model: str,
    eval_name: str,
    baseline: dict,
    patched: dict,
) -> None:
    print()
    print("=" * _W)
    print(f"  SGLang Quality: baseline vs fastokens ({eval_name})")
    print(f"  Model: {model}")
    print("=" * _W)
    print(f"  {'Metric':<32} {'Baseline':>12} {'Fastokens':>12} {'Change':>10}")
    print("-" * _W)

    b_score = baseline["score"]
    p_score = patched["score"]
    b_str = f"{b_score:.4f}"
    p_str = f"{p_score:.4f}"
    if b_score != 0:
        pct = (p_score - b_score) / b_score * 100
        change = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
    else:
        change = ""
    print(f"  {'Score':<32} {b_str:>12} {p_str:>12} {change:>10}")

    b_lat = baseline["latency_s"]
    p_lat = patched["latency_s"]
    b_str = f"{b_lat:.1f}s"
    p_str = f"{p_lat:.1f}s"
    if b_lat != 0:
        pct = (b_lat - p_lat) / b_lat * 100
        change = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
    else:
        change = ""
    print(f"  {'Eval latency':<32} {b_str:>12} {p_str:>12} {change:>10}")

    print("=" * _W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_at_double_dash(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main(argv: list[str] | None = None) -> None:
    our_argv, server_extra = _split_at_double_dash(
        argv if argv is not None else sys.argv[1:]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compare eval quality between fastokens-patched and stock "
            "SGLang tokenizer. Runs the same benchmark on both and "
            "compares scores."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s meta-llama/Llama-3.1-8B-Instruct
              %(prog)s meta-llama/Llama-3.1-8B-Instruct --eval gpqa --num-examples 50
              %(prog)s deepseek-ai/DeepSeek-V3 -- --tp 8
        """),
    )
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument(
        "--eval", dest="eval_name", default="gpqa",
        choices=["gpqa", "mmlu", "humaneval", "gsm8k", "mgsm"],
        help="Evaluation benchmark (default: gpqa)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None,
        help="Number of eval examples (default: all)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=64,
        help="Threads for parallel eval requests (default: 64)",
    )
    parser.add_argument(
        "--port", type=int, default=30000,
        help="Base port; baseline uses PORT, patched uses PORT+1 (default: 30000)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Server startup timeout in seconds (default: 600)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--baseline-only", action="store_true",
        help="Only run the baseline (unpatched) eval",
    )
    group.add_argument(
        "--patched-only", action="store_true",
        help="Only run the patched (fastokens) eval",
    )

    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results as JSON to this path",
    )

    args = parser.parse_args(our_argv)

    baseline_port = args.port
    patched_port = args.port + 1

    def _run_one(*, patched: bool, port: int) -> dict:
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

            print(
                f"  [{tag}] Running {args.eval_name} eval"
                f" ({args.num_examples or 'all'} examples, "
                f"{args.num_threads} threads)..."
            )
            result = _run_eval(
                base_url, args.model, args.eval_name,
                args.num_examples, args.num_threads,
            )

            print(f"  [{tag}] Score: {result['score']:.4f} "
                  f"({result['latency_s']:.1f}s)")
            return result
        finally:
            print(f"  [{tag}] Stopping server...")
            _stop_server(proc)

    baseline_result: dict | None = None
    patched_result: dict | None = None

    if not args.patched_only:
        baseline_result = _run_one(patched=False, port=baseline_port)

    if not args.baseline_only:
        patched_result = _run_one(patched=True, port=patched_port)

    if baseline_result and patched_result:
        _print_comparison(
            args.model, args.eval_name, baseline_result, patched_result,
        )

    if args.output:
        results: dict = {
            "model": args.model,
            "eval": args.eval_name,
            "num_examples": args.num_examples,
        }
        if baseline_result:
            results["baseline"] = {
                "score": baseline_result["score"],
                "latency_s": baseline_result["latency_s"],
            }
        if patched_result:
            results["fastokens"] = {
                "score": patched_result["score"],
                "latency_s": patched_result["latency_s"],
            }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
