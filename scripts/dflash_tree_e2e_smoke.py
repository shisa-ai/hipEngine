"""End-to-end DDTree verifier smoke against the full PARO target model.

This is the GPU correctness gate for task #12.  It loads the resident
target session, prefills a stable prompt, hand-builds a tree-shaped
``TargetVerifyBatch`` from arbitrary tokens (no drafter required), and
calls ``Qwen35ParoResidentSession.verify_tree_bulk_and_commit`` ONCE.

Gates checked:

  * The full forward (B+1 batched rows through all layer types, including
    the tree-aware GQA gate kernel on every full-attention layer) runs
    without crashing.
  * ``finite_logits == True`` (no NaN/Inf in the verifier logits).
  * ``gpu_accept_match_cpu == True`` (the existing chain accept summary
    kernel walks ``parent_rows`` correctly for tree topology).
  * The CPU oracle ``TargetVerifyBatch.accept_from_top1`` agrees with the
    GPU result.

Single-cycle only.  Multi-cycle decode requires the K/V compaction step
tracked in task #15.

Usage::

  HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
    python3 scripts/dflash_tree_e2e_smoke.py --backend hip_gfx1151 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from hipengine.benchmark.prompts import DEFAULT_STABLE_PROMPT_FIXTURE, load_prompt_records
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
)
from hipengine.speculative import DraftBatch, TargetAcceptSummary, TargetVerifyBatch


def _build_tree_batch(
    *,
    root_token: int,
    root_position: int,
    tree_tokens: Sequence[int],
    tree_parents: Sequence[int],
) -> TargetVerifyBatch:
    """Construct a tree-shaped ``TargetVerifyBatch`` from hand-built topology.

    ``tree_tokens[i]`` and ``tree_parents[i]`` describe candidate row ``i``
    (0-indexed in candidate space, NOT including the root).  ``tree_parents[i]``
    is ``-1`` if the parent is the root, otherwise a candidate index < ``i``.
    Draft depths are derived by following the parent chain.
    """

    if len(tree_tokens) != len(tree_parents):
        raise ValueError("tree_tokens and tree_parents must align")
    rows = len(tree_tokens)
    depths: list[int] = []
    parent_positions: list[int] = []
    for i in range(rows):
        parent = tree_parents[i]
        if parent < 0:
            depths.append(1)
            parent_positions.append(int(root_position))
        else:
            if parent >= i:
                raise ValueError(f"tree_parents[{i}]={parent} must be < {i}")
            depths.append(depths[parent] + 1)
            parent_positions.append(int(root_position) + depths[parent])
    draft = DraftBatch(
        request_ids=(0,),
        candidate_tokens=tuple(int(t) for t in tree_tokens),
        parent_positions=tuple(parent_positions),
        draft_depths=tuple(depths),
        row_to_request=(0,) * rows,
        tree_parents=tuple(int(p) for p in tree_parents),
        active_mask=(True,) * rows,
        mode="verify_tree",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(int(root_token),),
        root_positions=(int(root_position),),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e"))
    ap.add_argument("--backend", type=str, default="hip_gfx1151")
    ap.add_argument("--prompt-fixture", type=Path, default=DEFAULT_STABLE_PROMPT_FIXTURE)
    ap.add_argument("--prompt-index", type=int, default=0, help="Index into the prompt fixture to use as decode context")
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument(
        "--tree-shape",
        type=str,
        default="depth2_binary",
        choices=("depth2_binary", "depth1_branch4", "chain_reduction"),
        help="Synthetic tree topology to verify",
    )

    ap.add_argument("--json", type=Path, default=None, help="Write a one-row JSON artifact here")
    args = ap.parse_args()

    compiler_version: str | None = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip().splitlines()[0]
        if not compiler_version:
            compiler_version = None

    prompts = load_prompt_records(args.prompt_fixture)
    if args.prompt_index >= len(prompts):
        raise ValueError(f"prompt_index={args.prompt_index} out of range ({len(prompts)} prompts)")
    prompt = prompts[args.prompt_index]
    prompt_ids = [int(x) for x in prompt["prompt_ids"]]
    if not prompt_ids:
        raise ValueError("selected prompt has empty prompt_ids")

    # Tree topology definitions.  ``tree_tokens`` are chosen as arbitrary
    # vocabulary-valid ids in [1000, 1006]; the model's target_top1 will
    # likely not match these tokens, so we expect ``accepted_count == 0``
    # in most cases.  That's fine for a SMOKE test -- we are validating
    # finite outputs + GPU/CPU agreement, not acceptance quality.
    if args.tree_shape == "depth2_binary":
        # 6 candidates over a depth-2 binary tree:
        #   row 0 = root (committed)
        #   cand 0 = depth-1 child  (parent = root)
        #   cand 1 = depth-1 child  (parent = root)
        #   cand 2 = depth-2 child of cand 0
        #   cand 3 = depth-2 child of cand 0
        #   cand 4 = depth-2 child of cand 1
        #   cand 5 = depth-2 child of cand 1
        tree_tokens = (1000, 1001, 1002, 1003, 1004, 1005)
        tree_parents = (-1, -1, 0, 0, 1, 1)
    elif args.tree_shape == "depth1_branch4":
        # Single root with 4 depth-1 branches; smallest non-chain tree.
        tree_tokens = (1000, 1001, 1002, 1003)
        tree_parents = (-1, -1, -1, -1)
    elif args.tree_shape == "chain_reduction":
        # Degenerate tree that is a chain.  The tree verifier MUST handle
        # this and produce a result equivalent to the chain verifier.
        tree_tokens = (1000, 1001, 1002, 1003)
        tree_parents = (-1, 0, 1, 2)
    else:
        raise ValueError(f"unknown tree shape {args.tree_shape!r}")

    decode_position = len(prompt_ids)
    candidate_budget = len(tree_tokens)
    branch_factor = sum(1 for parent in tree_parents if parent < 0)
    print(
        f"[setup] backend={args.backend} model={args.model.name}\n"
        f"[setup] prompt_id={prompt.get('id')} prompt_tokens={len(prompt_ids)} "
        f"decode_position={decode_position}\n"
        f"[setup] tree_shape={args.tree_shape} tree_tokens={tree_tokens} "
        f"tree_parents={tree_parents} branches_at_root={branch_factor}"
    )

    runner = Qwen35ParoNextTokenRunner(args.model, backend=args.backend)
    max_sequence = decode_position + candidate_budget + 8
    max_batch_size = max(4, candidate_budget + 2)
    reset_memory_stats()
    session_kwargs: dict[str, Any] = dict(
        max_sequence_length=max_sequence,
        max_layers=args.max_layers,
        max_batch_size=max_batch_size,
        require_cached_build=args.require_cached_build,
        prefill_config=PrefillConfig(),
    )
    if compiler_version is not None:
        session_kwargs["compiler_version"] = compiler_version

    with Qwen35ParoResidentSession(runner, **session_kwargs) as session:
        # Prefill the prompt to slot 1 (slot 0 is conventional AR; tree
        # verifier doesn't care which slot we pick).
        t0 = time.perf_counter()
        next_result = None
        for pos, token in enumerate(prompt_ids):
            next_result = session.step(int(token), position=pos, sample=(pos == len(prompt_ids) - 1))
        if next_result is None:
            raise RuntimeError("prompt prefill produced no next-token result")
        prefill_seconds = time.perf_counter() - t0
        root_token = int(next_result.token_id)

        target_batch = _build_tree_batch(
            root_token=root_token,
            root_position=decode_position,
            tree_tokens=tree_tokens,
            tree_parents=tree_parents,
        )

        # ``capture_hidden_concat`` is a per-layer hidden tap output buffer
        # used by the drafter; tree-only smoke doesn't consume it but the
        # session entry validates shape, so allocate a small one matching
        # the verifier rows.
        capture_layers = (0,)  # arbitrary; we only check the tree forward's correctness
        from hipengine.core.memory import malloc
        from hipengine.core.tensor import Tensor
        from hipengine.core.dtype import DType

        capture_rows = target_batch.rows
        capture_width = len(capture_layers) * session.config.hidden_size
        capture_nbytes = capture_rows * capture_width * DType.BF16.itemsize
        capture_buf = malloc(capture_nbytes, runtime=session.runtime)
        capture_tensor = Tensor.from_handle(capture_buf.ptr, (capture_rows, capture_width), DType.BF16, session.device)

        # Verify ONCE.
        verify_base_slot = 1
        t_verify = time.perf_counter()
        verify_result = session.verify_tree_bulk_and_commit(
            target_batch,
            base_slot=verify_base_slot,
            capture_layer_ids=capture_layers,
            capture_hidden_concat=capture_tensor,
            capture_row_start=0,
        )
        verify_seconds = time.perf_counter() - t_verify

        # CPU oracle reproduction for ground truth.
        cpu_result = target_batch.accept_from_top1(verify_result.target_top1, transaction_id=0)
        cpu_summary = TargetAcceptSummary.from_accept_result(target_batch, cpu_result)

        gates = {
            "finite_logits": bool(verify_result.finite_logits),
            "gpu_accept_match_cpu": bool(verify_result.gpu_accept_match_cpu),
            "cpu_accepted_count_matches_verify_result": int(cpu_summary.accepted_counts[0]) == verify_result.accepted_count,
            "cpu_commit_row_matches_verify_result": int(cpu_summary.commit_rows[0]) == verify_result.commit_row,
        }

        # Multi-cycle sanity: after verify_tree compaction the cache MUST
        # still produce a finite next-token forward.  Broken K/V compaction
        # (sparse cache slots) would surface as NaN/Inf in this step.
        next_token = verify_result.next_token if verify_result.next_token is not None else verify_result.commit_token
        next_position = verify_result.commit_position + 1
        # The follow-up step always uses slot 0 -- it doesn't validate the
        # compacted cache directly (slot 0 wasn't touched by verify_tree),
        # but it confirms the session is in a usable state after compaction.
        # The proper multi-cycle compaction validation lives in the E2E
        # benchmark (task #14) once the tree drafter lands.
        followup = session.step(int(next_token), position=int(next_position), sample=True)
        followup_token: int | None = None
        followup_logit_finite = False
        if followup is not None:
            followup_token = int(followup.token_id)
            followup_logit_finite = math.isfinite(float(followup.logit))
        gates["followup_decode_finite"] = bool(followup_logit_finite)
        gates["followup_decode_emitted_token"] = followup_token is not None

        all_passed = all(gates.values())

        report = {
            "backend": session.backend,
            "target_arch": session.target_arch,
            "model": str(args.model.name),
            "tree_shape": args.tree_shape,
            "prompt_id": prompt.get("id"),
            "prompt_tokens": len(prompt_ids),
            "root_token": root_token,
            "decode_position": decode_position,
            "tree_tokens": list(tree_tokens),
            "tree_parents": list(tree_parents),
            "rows": target_batch.rows,
            "verify_result": {
                "target_top1": list(verify_result.target_top1),
                "accepted_count": verify_result.accepted_count,
                "commit_row": verify_result.commit_row,
                "commit_token": verify_result.commit_token,
                "commit_position": verify_result.commit_position,
                "next_token": verify_result.next_token,
                "full_accept": verify_result.full_accept,
                "finite_logits": verify_result.finite_logits,
                "gpu_accept_match_cpu": verify_result.gpu_accept_match_cpu,
            },
            "cpu_oracle": {
                "accepted_count": int(cpu_summary.accepted_counts[0]),
                "commit_row": int(cpu_summary.commit_rows[0]),
                "commit_token": int(cpu_summary.commit_tokens[0]),
                "commit_position": int(cpu_summary.commit_positions[0]),
                "next_token": None if cpu_summary.next_tokens is None else int(cpu_summary.next_tokens[0]),
                "full_accept": bool(cpu_summary.full_accept[0]),
            },
            "gates": gates,
            "all_correctness_passed": all_passed,
            "prefill_seconds": prefill_seconds,
            "verify_seconds": verify_seconds,
            "verify_base_slot": verify_base_slot,
            "followup": {
                "input_token": int(next_token),
                "input_position": int(next_position),
                "emitted_token": followup_token,
                "logit_finite": followup_logit_finite,
            },
            "memory": memory_stats(),
        }

        print(json.dumps(report, indent=2))
        if args.json is not None:
            args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if not all_passed:
            print(f"FAIL: gates {gates}", file=sys.stderr)
            return 1
        print(f"[OK] DDTree verifier smoke ({args.tree_shape}): all correctness gates passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
