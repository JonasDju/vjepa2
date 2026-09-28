# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in diagnostics for the Inductor ``CantSplit`` crash under ``compile_model: true``.

Set ``VJEPA_DEBUG_CANTSPLIT=1`` in the job environment. When Inductor fails to split a fused
kernel's iteration ranges, this logs the kernel's iteration groups and, for every node fused into
it, the node's ranges, the aten ops it came from and the model source lines they were traced from
-- then re-raises the original error unchanged. Nothing else about compilation changes.

Uses private Inductor internals (torch 2.13); if they move, installation logs a warning and does
nothing instead of breaking training.
"""

import logging

logger = logging.getLogger(__name__)


def _origin_summary(node, max_origins=6, max_trace_lines=4):
    ir_node = getattr(node, "node", None)
    origins = list(getattr(ir_node, "origins", None) or [])[:max_origins]
    lines = []
    for o in origins:
        lines.append(f"      origin {o.name}: {getattr(o, 'target', '?')}")
        trace = o.meta.get("stack_trace") or ""
        # drop the ^^^^ / ~~~~ caret lines Python adds under each source line
        frames = [ln.strip() for ln in trace.splitlines() if ln.strip() and set(ln.strip()) - set("^~ ")]
        for ln in frames[-max_trace_lines:]:
            lines.append(f"        | {ln}")
    return lines


def _describe(node_schedule, kernel, failed_ranges):
    out = [f"  kernel groups (numels): {dict(getattr(kernel, 'numels', {}))}"]
    for node in node_schedule:
        if not hasattr(node, "get_ranges"):  # EnableReduction / DisableReduction markers
            out.append(f"  <{getattr(node, '__name__', node)}>")
            continue
        failed = " <-- ranges of the failed split" if failed_ranges is not None and node.get_ranges() == failed_ranges else ""
        out.append(f"  node {node.get_name()}: ranges={node.get_ranges()}{failed}")
        out.extend(_origin_summary(node))
    return "\n".join(out)


def install_cantsplit_logger(rank=0):
    """Wrap ``SIMDScheduling.codegen_node_schedule_with_kernel`` to log the failing kernel."""
    try:
        from torch._inductor.codegen.simd import CantSplit, SIMDScheduling
    except ImportError as e:  # pragma: no cover - depends on the torch version
        logger.warning(f"CantSplit logger not installed (torch internals moved: {e})")
        return False

    original = SIMDScheduling.codegen_node_schedule_with_kernel
    if getattr(original, "_vjepa_cantsplit_logger", False):
        return True

    def wrapped(self, node_schedule, kernel):
        attempted = []
        split = kernel.split_and_set_ranges

        def recording_split(lengths):
            # called with node.get_ranges() for every node, in two passes over the schedule
            attempted.append(lengths)
            return split(lengths)

        kernel.split_and_set_ranges = recording_split
        try:
            return original(self, node_schedule, kernel)
        except CantSplit as e:
            failed_ranges = attempted[-1] if attempted else None
            logger.error(
                f"[rank {rank}] Inductor CantSplit: {getattr(e, 'expr', '?')} not divisible by "
                f"{getattr(e, 'remaining', '?')}\n{_describe(node_schedule, kernel, failed_ranges)}"
            )
            raise
        finally:
            del kernel.split_and_set_ranges

    wrapped._vjepa_cantsplit_logger = True
    SIMDScheduling.codegen_node_schedule_with_kernel = wrapped
    logger.info("Inductor CantSplit logger installed (VJEPA_DEBUG_CANTSPLIT)")
    return True
