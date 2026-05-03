"""Eval harness: runs (method × benchmark × length) grid, prints % accuracy.

Usage:
    python3 eval.py                # default settings
    python3 eval.py --quick        # minimal smoke run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

from benchmarks import oolong_lite, sniah
from rlm.baseline import run_baseline
from rlm.client import LLMClient
from rlm.rlm import run_rlm


@dataclass
class CellResult:
    n: int
    n_correct: int
    n_errors: int
    elapsed_s: float

    @property
    def pct(self) -> float:
        return 100.0 * self.n_correct / max(1, self.n)


def _fmt(cell: CellResult) -> str:
    if cell.n == 0:
        return "    -    "
    err = f" (e{cell.n_errors})" if cell.n_errors else ""
    return f"{cell.pct:5.1f}%{err}"


def run_cell(
    bench_name: str,
    method: str,
    target_chars: int,
    n_instances: int,
    seed_base: int,
    root_client: LLMClient,
    sub_client: LLMClient,
) -> CellResult:
    n_correct = 0
    n_errors = 0
    t0 = time.time()
    for i in range(n_instances):
        seed = seed_base + i
        if bench_name == "sniah":
            inst = sniah.make_instance(target_chars, seed)
            scorer = lambda pred: sniah.score(pred, inst.gold)
        else:
            inst = oolong_lite.make_instance(target_chars, seed)
            scorer = lambda pred: oolong_lite.score(pred, inst.gold)

        try:
            if method == "baseline":
                res = run_baseline(inst.query, inst.context, root_client)
                pred = res.answer
            elif method == "rlm":
                res = run_rlm(
                    inst.query,
                    inst.context,
                    root_client=root_client,
                    sub_client=sub_client,
                    max_iterations=8,
                )
                pred = res.answer
            elif method == "rlm-no-sub":
                res = run_rlm(
                    inst.query,
                    inst.context,
                    root_client=root_client,
                    sub_client=None,
                    max_iterations=8,
                )
                pred = res.answer
            else:
                raise ValueError(method)
            ok = scorer(pred)
            n_correct += int(ok)
        except Exception as e:
            n_errors += 1
            print(f"      [err] {bench_name}/{method}/{target_chars}/seed={seed}: {e}", file=sys.stderr)
        sys.stdout.write(".")
        sys.stdout.flush()
    return CellResult(n=n_instances, n_correct=n_correct, n_errors=n_errors, elapsed_s=time.time() - t0)


def print_table(bench: str, lengths: list[int], methods: list[str], grid: dict) -> None:
    print(f"\n=== {bench.upper()} — accuracy % (n per cell shown in header) ===")
    header = f"{'method':<14}" + "".join(f"  {L:>6}c" for L in lengths)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<14}"
        for L in lengths:
            row += "  " + _fmt(grid[(m, L)])
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Tiny grid for smoke test")
    ap.add_argument("--bench", choices=["sniah", "oolong", "both"], default="both")
    ap.add_argument("--n", type=int, default=None, help="instances per cell")
    args = ap.parse_args()

    if args.quick:
        lengths = [2_000, 8_000]
        n = args.n or 3
        methods = ["baseline", "rlm-no-sub", "rlm"]
    else:
        lengths = [4_000, 16_000, 64_000]
        n = args.n or 8
        methods = ["baseline", "rlm-no-sub", "rlm"]

    root_model = os.environ.get("RLM_ROOT_MODEL", "anthropic/claude-haiku-4.5")
    sub_model = os.environ.get("RLM_SUB_MODEL", "anthropic/claude-haiku-4.5")
    print(f"root model: {root_model}")
    print(f"sub  model: {sub_model}")
    print(f"lengths: {lengths}   n per cell: {n}   methods: {methods}")

    benches = ["sniah", "oolong"] if args.bench == "both" else [args.bench]

    overall_t0 = time.time()
    for bench in benches:
        print(f"\n>>> running {bench}")
        grid: dict = {}
        for m in methods:
            for L in lengths:
                # fresh clients per cell so token totals are per-cell
                root_client = LLMClient(model=root_model)
                sub_client = LLMClient(model=sub_model)
                print(f"\n  [{bench}/{m}/{L}c] ", end="")
                grid[(m, L)] = run_cell(bench, m, L, n, seed_base=hash((bench, L)) & 0xFFFF, root_client=root_client, sub_client=sub_client)
                cell = grid[(m, L)]
                print(f" {cell.pct:.1f}%  ({cell.elapsed_s:.1f}s)")
        print_table(bench, lengths, methods, grid)

    print(f"\nTotal time: {time.time() - overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
