# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare the gfx950 MX GEMM formats: a6w4 vs a6w6, a8w4, a4w4.

Every path starts from the *same* bf16 tensors and then runs its own canonical
aiter quantization pipeline, so the comparison is between formats rather than
between quantizers:

  * a4w4 / a6w4 / a8w4 (FlyDSL) share one kernel and differ only in ``a_dtype``,
    which isolates activation precision exactly at fixed MXFP4 weights.
  * a4w4 (asm) and a6w6 (asm) are aiter's production paths and act as the
    reality check on the FlyDSL arm.

Tuning: the asm paths auto-select a shape-tuned kernel; the FlyDSL arm uses
``pick_mx_tiles``, which measured within ~9% of an exhaustive tile sweep on
these shapes. Pass ``--sweep`` to tile-sweep the FlyDSL arm instead.

Note the inputs are Gaussian, which is the *best* case for low-bit weights --
real LLM tensors have outliers that MXFP4 handles far worse than MXFP6, so the
a4w4-vs-a6w4 accuracy gap here is a lower bound.

Usage:
    python op_tests/op_benchmarks/hip/bench_gemm_mx_a6w4.py [--sweep]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import torch

DEFAULT_SHAPES = [
    (1, 4096, 4096),
    (128, 4096, 4096),
    (1024, 4096, 4096),
    (4096, 4096, 4096),
    (2048, 8192, 8192),
]
SWEEP_TILES = list(itertools.product((32, 64, 128, 256), (64, 128, 256), (128, 256)))


def _bench(fn, iters: int = 100, warmup: int = 25) -> float:
    """Microseconds per call: min over 3 reps of the mean, to suppress noise."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
    return best


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(out.float() - ref) / torch.linalg.vector_norm(ref)
    )


def _flydsl_entries(x, w, N, ref, sweep):
    from aiter.ops.flydsl.batched_gemm_mxfp4 import (
        flydsl_gemm_mxfp4,
        preshuffle_mx_weight,
        quant_mx_act,
        tiles_are_valid,
    )

    _, K = x.shape
    w_codes, w_scales = preshuffle_mx_weight(w)
    out = []
    for a_dtype, name in (("fp4", "a4w4-flydsl"), ("fp6", "a6w4-flydsl"),
                          ("fp8", "a8w4-flydsl")):
        try:
            a_codes, a_scales = quant_mx_act(x, a_dtype)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: quant unavailable ({exc})", file=sys.stderr)
            continue

        def call(tiles=None, a_codes=a_codes, a_scales=a_scales, a_dtype=a_dtype):
            kw = {} if tiles is None else dict(
                tile_m=tiles[0], tile_n=tiles[1], tile_k=tiles[2]
            )
            return flydsl_gemm_mxfp4(a_codes, w_codes, a_scales, w_scales, N,
                                     torch.bfloat16, a_dtype=a_dtype, **kw)

        if not sweep:
            out.append((name, _bench(call), _rel(call(), ref), "pick_mx_tiles"))
            continue
        best = (float("inf"), None)
        for tiles in SWEEP_TILES:
            if not tiles_are_valid(*tiles, N, K, a_dtype):
                continue
            try:
                call(tiles)
                torch.cuda.synchronize()
                us = _bench(lambda tiles=tiles: call(tiles))
            except Exception:  # noqa: BLE001
                continue
            best = min(best, (us, tiles))
        if best[1]:
            out.append((name, best[0], _rel(call(best[1]), ref), str(best[1])))
    return out


def _asm_entries(x, w, M, N, K, ref):
    out = []
    try:
        from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

        A, As = quant_mxfp6_gemm(x)
        B, Bs = quant_mxfp6_gemm(w)
        run = lambda: gemm_a6w6(A, B, As, Bs, M, N, K)  # noqa: E731
        out.append(("a6w6-asm", _bench(run), _rel(run(), ref), "auto"))
    except Exception as exc:  # noqa: BLE001
        print(f"  a6w6-asm unavailable: {exc}", file=sys.stderr)

    try:
        import aiter
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        from aiter.ops.shuffle import shuffle_weight

        qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        xq, xs = qf(x, shuffle=True)
        wq, ws = qf(w, shuffle=True)
        wsh = shuffle_weight(wq, layout=(16, 16))
        run = lambda: gemm_a4w4(xq, wsh, xs, ws, bpreshuffle=True)  # noqa: E731
        got = run()
        got = got[0] if isinstance(got, tuple) else got
        out.append(("a4w4-asm", _bench(run), _rel(got[:M], ref), "auto"))
    except Exception as exc:  # noqa: BLE001
        print(f"  a4w4-asm unavailable: {exc}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true",
                    help="tile-sweep the FlyDSL arm instead of using pick_mx_tiles")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        print(f"this benchmark is gfx950-only, got {arch}", file=sys.stderr)
        return 1

    for M, N, K in DEFAULT_SHAPES:
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.5
        ref = x.float() @ w.float().T
        flops = 2 * M * N * K

        entries = _flydsl_entries(x, w, N, ref, args.sweep)
        entries += _asm_entries(x, w, M, N, K, ref)
        run = lambda: x @ w.T  # noqa: E731
        entries.append(("bf16", _bench(run), 0.0, "-"))

        print(f"\n=== M={M} N={N} K={K} ===")
        print(f"{'path':<14} {'us':>9} {'TFLOP/s':>9} {'rel err':>10}  config")
        for name, us, rel, cfg in sorted(entries, key=lambda e: e[1]):
            print(f"{name:<14} {us:9.1f} {flops / (us / 1e6) / 1e12:9.1f} "
                  f"{rel:10.3e}  {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
