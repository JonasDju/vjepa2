# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Workaround for an Inductor bug that crashes ``compile_model: true`` (torch 2.13).

The predictor's token axis is ``n_ctxt + n_pred`` -- the sum of two dynamic sizes once the first
recompile makes them symbolic. Inductor fuses the residual adds into the next block's ``norm1``
LayerNorm and must prove that the flattened residual node (``18432*s3 + 18432*s89`` elements, batch
48 x dim 384) splits into the LayerNorm's groups (``48*s3 + 48*s89`` rows x 384).
``SizeVarAllocator.statically_known_multiple_of`` cannot see through the sum (it builds
``Mod(18432*s3 + 18432*s89, 48*s3 + 48*s89)``, which sympy does not simplify) and the codegen raises
``InductorError: CantSplit``.

The patch keeps Inductor's answer whenever it is already ``True`` and otherwise tries exact rational
simplification: ``numerator`` is a multiple of ``denominator`` if ``sympy.cancel(numerator /
denominator)`` is an integer-valued expression (here exactly ``384``). That is a proof, not a
heuristic -- a quotient that only *might* be integral (``is_integer`` is ``None``) is rejected -- so
Inductor never emits wrong indexing because of it. It only lets Inductor accept a split it would
otherwise give up on; the generated kernels compute the same thing.

Verified not to change what the model computes
----------------------------------------------
Compiling cannot be bit-identical to eager (Inductor fuses ops and keeps intermediates in fp32, so the
rounding differs), so "same behaviour" means: the compiled training step is no less accurate than the
eager one. Checked on the cluster (H100, torch 2.13, 2026-09-29) with the CUDA-only test
``tests/vjepa_2_1/test_compile_equivalence_cuda.py`` (kept on branch ``debug-predictor-compile``; too heavy
for the CPU dev box). It builds and compiles the model exactly like train.py (``compile_models`` below,
this patch included), feeds the same batch through the eager and the compiled model, and compares the
target-encoder output, both predictor outputs, all three losses and every parameter gradient against a
higher-precision eager reference, over 3 steps with changing mask shapes (the second step is the
dynamic-shape recompile that used to crash). Trained weights: the 4-GPU run's epoch-60 checkpoint.

- float32 (TF32 off) vs a float64 reference: compiled and eager errors are the same size -- outputs
  1.18e-6 vs 1.19e-6, the whole gradient vector 8.8e-6 vs 8.3e-6, losses equal to three digits.
- the training setting (bfloat16 autocast, batch 48, ``activation_memory_budget`` 0.5) vs a float32
  reference: compiled is slightly *more* accurate than eager -- target-encoder output 4.6e-3 vs 7.1e-3,
  predictor outputs 3.5-4.8e-3 vs 4.2-5.3e-3, whole gradient vector 4.4-4.9e-3 vs 5.0-5.2e-3.

A defect in the compiled graph (wrong indexing, a dropped term, bad recomputation) gives errors of order 1;
a 0.1 % compile-only perturbation injected in a CPU dry run exceeded the eager error by a median factor of
~400 (outputs) to ~3600 (gradients) and failed 1023 comparisons.
Scope of that verification: that checkpoint predates the bounded-QK options (``qk_norm_affine: false``,
``qk_temperature_max``), so they were not part of it; ``activation_memory_budget`` only moves the
store/recompute boundary. The predictor's gather-based token reorder (``predictor.reorder_tokens``) is
bit-identical to the loop it replaces (tests/vjepa_2_1/test_predictor_reorder.py).
"""

import logging

import sympy
import torch

logger = logging.getLogger(__name__)

# same guard as Inductor's own sympy fallback (sizevars._MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS in torch 2.13)
_MAX_SYMBOLS = 20


def exact_multiple_of(numerator, denominator):
    """True iff ``numerator / denominator`` cancels to an expression sympy knows is an integer."""
    numerator, denominator = sympy.sympify(numerator), sympy.sympify(denominator)
    if denominator == 0 or len(numerator.free_symbols | denominator.free_symbols) > _MAX_SYMBOLS:
        return False
    try:
        quotient = sympy.cancel(numerator / denominator)
    except (TypeError, ValueError, sympy.PolynomialError):
        return False
    return quotient.is_integer is True


def patch_inductor_multiple_of():
    """Install the fallback on ``SizeVarAllocator.statically_known_multiple_of`` (idempotent)."""
    try:
        from torch._inductor.sizevars import SizeVarAllocator
    except ImportError as e:  # pragma: no cover - depends on the torch version
        logger.warning(f"Inductor multiple-of fallback not installed (torch internals moved: {e})")
        return False

    original = SizeVarAllocator.statically_known_multiple_of
    if getattr(original, "_vjepa_cancel_fallback", False):
        return True

    def statically_known_multiple_of(self, numerator, denominator):
        if original(self, numerator, denominator):
            return True
        if isinstance(denominator, (int, sympy.Integer)):
            return False  # Inductor's own integer-denominator check is already exact
        return exact_multiple_of(numerator, denominator)

    statically_known_multiple_of._vjepa_cancel_fallback = True
    statically_known_multiple_of._vjepa_original = original
    SizeVarAllocator.statically_known_multiple_of = statically_known_multiple_of
    logger.info("Inductor statically_known_multiple_of: installed exact sympy.cancel fallback (CantSplit workaround)")
    return True


def compile_models(encoder, target_encoder, predictor, activation_memory_budget=None):
    """``compile_model: true`` as train.py does it.

    ``activation_memory_budget`` (None = leave PyTorch's default) is only applied without activation
    checkpointing; the caller decides. The CUDA-only check that this compiled model computes the same training
    step as the uncompiled one (tests/vjepa_2_1/test_compile_equivalence_cuda.py) and the Inductor CantSplit
    diagnostics (app/vjepa_2_1/compile_debug.py, VJEPA_DEBUG_CANTSPLIT) live on branch debug-predictor-compile.
    """
    torch._dynamo.config.optimize_ddp = False
    patch_inductor_multiple_of()  # Inductor CantSplit on the predictor's n_ctxt + n_pred token axis
    encoder.compile()
    target_encoder.compile()
    predictor.compile()
    if activation_memory_budget is not None:
        torch._functorch.config.activation_memory_budget = activation_memory_budget
