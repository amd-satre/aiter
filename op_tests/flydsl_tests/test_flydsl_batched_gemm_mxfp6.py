# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the gfx950 MXFP A x MXFP4 B preshuffle GEMM via
``flydsl_batched_gemm_mxfp4``.

Covers ``a_dtype="fp4"`` (the pre-existing path) and ``a_dtype="fp6"``, which
the vendored ``kernels/mxfp4_preshuffle.py`` enables.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flydsl")
from aiter.ops.flydsl import is_flydsl_available
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return False
    return arch.lower().split(":")[0] == "gfx950"


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="the MXFP preshuffle GEMM path is gfx950/CDNA4 only",
)


# B > 1 is deliberately not covered. The batched A-scale layout is caller-supplied
# via sca_row_stride / sca_batch_stride, and aiter has no helper that produces it --
# flydsl_batched_gemm_mxfp4's docstring points at a `preshuffle_operands` that is not
# defined anywhere in the tree. Asserting a guessed layout would be worse than not
# testing it; single-batch is enough to cover the vendored kernel and the call site.
@pytest.mark.parametrize("a_dtype", ["fp4", "fp6"])
@pytest.mark.parametrize("B", [1])
def test_flydsl_batched_gemm_mxfp4_a_dtypes(a_dtype, B):
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
    from aiter.ops.flydsl.mxfp6_utils import (
        fp6_e2m3_to_f32,
        per_1x32_f6_quant,
        shuffle_scale_w4,
    )
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight_NK

    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N, K = 128, 1024, 8192

    a_bf = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    w_bf = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # A operand.
    if a_dtype == "fp6":
        a_codes, a_scale, a_unpacked = per_1x32_f6_quant(a_bf.view(-1, K))
        a_deq = fp6_e2m3_to_f32(a_unpacked)
    else:
        a_codes, a_scale = per_1x32_f4_quant(a_bf.view(-1, K).float())[:2]
        a_deq = mxfp4_to_f32(a_codes)
    a_deq = a_deq * e8m0_to_f32(a_scale.repeat_interleave(32, dim=1))

    # B operand: MXFP4, CK-preshuffled. shuffle_weight_NK(w, 16, 64) is the same
    # layout the kernel expects; only the scale needs the CDNA4-specific shuffle.
    w_q, w_scale = per_1x32_f4_quant(w_bf.float())[:2]
    w_deq = mxfp4_to_f32(w_q) * e8m0_to_f32(w_scale.repeat_interleave(32, dim=1))

    out = flydsl_batched_gemm_mxfp4(
        a_codes.view(B, M, -1),
        shuffle_weight_NK(w_q, 16, 64),
        shuffle_scale_w4(a_scale, 1, False),
        shuffle_scale_w4(w_scale, 1, False),
        N,
        torch.bfloat16,
        a_dtype=a_dtype,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    assert tuple(out.shape) == (B, M, N)

    # Reference is the dequantised operands, so this isolates GEMM accumulation
    # error from the (much larger) quantization error.
    ref = a_deq @ w_deq.T
    rel = torch.linalg.vector_norm(
        out.reshape(-1, N).float() - ref
    ) / torch.linalg.vector_norm(ref)
    assert rel < 1e-2, f"relative error {rel:.4e} too high (a_dtype={a_dtype}, B={B})"
