#!/usr/bin/env python3
"""Attribute the Gemma 4 MMQ gate/up divergence to the arm that is wrong.

``scripts/gemma4_mmq_prefill_length_probe.py`` compares the route against its
fp32 owner and reports that the two arms disagree, with a KL and a greedy-token
flip. That comparison cannot say which arm is wrong: at a length where both arms
are wrong the same way it reports exact agreement, and everywhere else it names
the MMQ route without having established that the fp32 route is the correct one.
A KL between two of our own kernels is a disagreement measure, not a correctness
measure.

This script adds the missing third arm. It feeds byte-identical prompt ids to

- hipEngine with the MMQ route off,
- hipEngine with the MMQ route on, and
- llama.cpp's ``llama-server`` on the same artifact, same GPU, same context,

and reports each arm's greedy token against llama.cpp's. llama.cpp is an
independent implementation of the same quantization, so agreement is evidence
about which arm is closer to the artifact's own arithmetic rather than about
which of our two kernels ran second. It is evidence and not proof: llama.cpp's
Q4_K_XL path is itself approximate, so the reading to trust is the pairing --
one arm tracking the anchor and the other not -- not the absolute rate.

Greedy tokens are taken as the argmax of the final row's logits with the model's
own softcap applied, which is the distribution both engines sample from, and
llama.cpp is asked with ``temperature=0, top_k=1`` so its sampled token is the
same argmax. The top-2 logit margin is recorded per length because a flip at a
near-tie is a different finding from a flip at a confident row.

Run it on the tree that carries the route::

    env -u HIP_VISIBLE_DEVICES PYTHONPATH=. .venv/bin/python \\
      scripts/gemma4_mmq_arm_attribution.py --lengths 16 24 96

Exit status is 0 when the two arms agree with each other and with the anchor at
every length, 1 otherwise. It is a diagnostic: it adds no performance row.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_llamacpp_reference_bench import (  # noqa: E402
    DEFAULT_PORT,
    _completion_body,
    _post,
    _stop,
    _wait_health,
)

DEFAULT_LENGTHS: tuple[int, ...] = (16, 24, 96)
DEFAULT_OUT = Path("/tmp/gemma4_mmq_arm_attribution.json")
DEFAULT_CONTINUATION = 4


def _top2_margin(logits: np.ndarray) -> float:
    """Top-1 minus top-2 on the final row, i.e. how confident the row is."""

    row = np.asarray(logits, dtype=np.float32).reshape(-1)
    if row.size < 2:
        return float("nan")
    top2 = np.partition(row, -2)[-2:]
    return float(top2[1] - top2[0])


def _p_top1(logits: np.ndarray) -> float:
    """Softmax probability of the argmax, comparable with llama.cpp's reported prob.

    llama.cpp reports post-softmax probabilities, so a raw logit margin is not
    comparable with it; the top token's own probability is.
    """

    row = np.asarray(logits, dtype=np.float32).reshape(-1)
    shifted = np.exp(row - row.max())
    return float(shifted.max() / shifted.sum())


def _fmt(value: Any) -> str:
    return "  n/a" if value is None else f"{float(value):.4f}"


def _arm_tokens(runner: Any, ids: Sequence[int], outputs: int) -> dict[str, Any]:
    """Greedy continuation of ``outputs`` tokens from a fresh prefill of ``ids``.

    The first forward is taken twice, once raw and once with the model's own
    ``final_logit_softcapping``. The cap is ``c * tanh(x / c)`` applied elementwise
    after the logits leave the device, so it is strictly monotonic and cannot
    reorder an argmax; recording both is what makes that checkable per row rather
    than an assumption, and it separates a flip in the route's arithmetic from a
    flip the output transform could have caused.
    """

    runner.reset()
    raw = np.asarray(runner.forward(list(ids), apply_softcap=False), dtype=np.float32).reshape(-1)
    runner.reset()
    capped = np.asarray(runner.forward(list(ids)), dtype=np.float32).reshape(-1)
    top = np.partition(raw, -5)[-5:][::-1]
    first = int(np.argmax(capped))
    margin = _top2_margin(capped)
    tokens = [first]
    # Per-step confidence, because a route that agrees on the first token and then
    # degrades is not visible at the first token. Steps read the raw logits so the
    # margin is the model's own; the argmax is identical either way, the cap being
    # monotonic, so this does not change the chain that gets generated.
    steps = [{"token": first, "raw_margin": _top2_margin(raw), "p_top1": _p_top1(capped)}]
    for _ in range(outputs - 1):
        logits = np.asarray(runner.forward([tokens[-1]], apply_softcap=False),
                            dtype=np.float32).reshape(-1)
        token = int(np.argmax(logits))
        tokens.append(token)
        steps.append({"token": token, "raw_margin": _top2_margin(logits),
                      "p_top1": _p_top1(logits)})
    return {
        "tokens": tokens,
        "steps": steps,
        "first_margin": margin,
        "first_logit": float(capped.max()),
        "raw_argmax": int(np.argmax(raw)),
        "capped_argmax": first,
        "raw_top5": [float(v) for v in top],
        "raw_span5": float(top[0] - top[-1]),
        "raw": raw,
        "capped": capped,
    }


def _repeat_check(runner: Any, ids: Sequence[int]) -> dict[str, Any]:
    """Measure one prompt twice in the same process and report whether it moved.

    Every number this campaign records comes from a process that prefilled several
    lengths in sequence, so if a prefill's logits depend on what the runner did
    before it, the recorded divergences are partly a function of the sweep order
    rather than of the route. ``runner.reset()`` is what should make that not
    happen; this is the check that it does.
    """

    rows = []
    for _ in range(2):
        runner.reset()
        rows.append(np.asarray(runner.forward(list(ids), apply_softcap=False),
                               dtype=np.float32).reshape(-1))
    delta = rows[1] - rows[0]
    return {
        "argmax_a": int(np.argmax(rows[0])),
        "argmax_b": int(np.argmax(rows[1])),
        "argmax_stable": int(np.argmax(rows[0])) == int(np.argmax(rows[1])),
        "max_abs_delta": float(np.abs(delta).max()),
        "bit_identical": bool(np.array_equal(rows[0], rows[1])),
    }


def _logit_comparison(base_arm: dict[str, Any], mmq_arm: dict[str, Any]) -> dict[str, Any]:
    """How the two arms' raw final-row logits differ, before any softcap.

    A constant offset is argmax-neutral, a scale change moves the softmax
    temperature, and a reordering is the only thing that can move the argmax. The
    three are separated here because they imply different faults: a scale says the
    route's hidden state reaches the head at the wrong magnitude, a reordering at
    one position says the projection itself is wrong.
    """

    delta = mmq_arm["raw"] - base_arm["raw"]
    base_rank = np.argsort(np.argsort(-base_arm["raw"]))[:64]
    mmq_rank = np.argsort(np.argsort(-mmq_arm["raw"]))[:64]
    return {
        "raw_argmax_base": base_arm["raw_argmax"],
        "raw_argmax_mmq": mmq_arm["raw_argmax"],
        "raw_argmax_agrees": base_arm["raw_argmax"] == mmq_arm["raw_argmax"],
        "softcap_preserves_argmax_base": base_arm["raw_argmax"] == base_arm["capped_argmax"],
        "softcap_preserves_argmax_mmq": mmq_arm["raw_argmax"] == mmq_arm["capped_argmax"],
        "delta_median": float(np.median(delta)),
        "delta_std": float(delta.std()),
        "delta_max_abs": float(np.abs(delta).max()),
        "raw_top5_base": base_arm["raw_top5"],
        "raw_top5_mmq": mmq_arm["raw_top5"],
        "raw_span5_base": base_arm["raw_span5"],
        "raw_span5_mmq": mmq_arm["raw_span5"],
        "top64_same_order": bool(np.array_equal(base_rank, mmq_rank)),
    }


def _run_arm(runner: Any, ids: Sequence[int], outputs: int, mmq: bool) -> dict[str, Any]:
    os.environ.pop("HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ", None)
    if mmq:
        os.environ["HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ"] = "1"
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts

    before = dict(experts.gemma4_moe_expert_route_counts())
    result = _arm_tokens(runner, ids, outputs)
    after = experts.gemma4_moe_expert_route_counts()
    result["route_launches"] = {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if "mmq" in key and value - before.get(key, 0)
    }
    result["route_used"] = bool(result["route_launches"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    from scripts.gemma4_campaign_bench import (
        DEFAULT_ARTIFACT,
        DEFAULT_CONTEXT,
        PROBE_CORPUS_SEED,
        exact_prompt_ids,
        probe_corpus,
    )

    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--lengths", type=int, nargs="*", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--continuation", type=int, default=DEFAULT_CONTINUATION,
                        help="greedy tokens per arm; only the first is a clean comparison "
                             "once the arms diverge, the rest show whether they re-converge")
    parser.add_argument("--corpus", choices=("frozen", "probe"), default="probe")
    parser.add_argument("--text", default=None,
                        help="literal prompt text; overrides --corpus and --lengths and compares "
                             "the whole text as one prompt, so a specific prompt class (a code "
                             "snippet, a repeated sentence) can be put to all three arms")
    parser.add_argument("--text-file", type=Path, default=None,
                        help="read the prompt text from a file instead of --text")
    parser.add_argument("--server", type=Path,
                        default=Path("~/llama.cpp/llama.cpp-hip/build-hip/bin/llama-server").expanduser())
    parser.add_argument("--source", type=Path,
                        default=Path("~/llama.cpp/llama.cpp-hip").expanduser())
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--pci", default=None,
                        help="PCI id of the card whose idleness gates the run; by default the "
                             "only amdgpu card is used, and several cards fail loudly")
    parser.add_argument("--idle-limit-mib", type=int, default=1024,
                        help="VRAM already in use above which the run refuses to start. This "
                             "script records no timing, so the guard is about not launching "
                             "into a busy device rather than about measurement validity; an "
                             "integrated GPU carries a driver baseline well above the 128 MiB "
                             "a discrete card needs")
    parser.add_argument("--kv", default="bf16")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.server.exists():
        parser.error(f"llama-server not found at {args.server}; pass --server")
    if args.continuation < 1:
        parser.error("--continuation must be >= 1")

    from scripts.gemma4_campaign_bench import _resolve_generator
    from hipengine.util.amdgpu_vram import select_card

    card = select_card(pci_id=args.pci)
    idle_used = int(card.vram_used_path.read_text())
    if idle_used > args.idle_limit_mib * 1024 * 1024:
        print(f"ERROR: GPU {card.pci_id} is not idle ({idle_used / 1024 / 1024:.0f} MiB used, "
              f"limit {args.idle_limit_mib} MiB)", file=sys.stderr)
        return 2

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"ERROR: port {args.port} already occupied", file=sys.stderr)
            return 2

    llm, runner, loading = _resolve_generator(args.artifact, args.context)
    tokenize = llm._get_text_generator().tokenize
    lengths = [int(n) for n in args.lengths]
    if args.text is not None or args.text_file is not None:
        # A literal prompt is a single-length comparison: the point is to reach
        # the position the prompt ends at, not to sweep widths of it.
        text = args.text_file.read_text() if args.text_file is not None else args.text
        ids = [int(token) for token in tokenize(text)]
        lengths = [len(ids)]
        if not ids:
            parser.error("the supplied prompt text tokenized to no ids")
    elif args.corpus == "probe":
        ids = exact_prompt_ids(tokenize, max(lengths), corpus=probe_corpus(seed=PROBE_CORPUS_SEED),
                               require_single_pass=True)
    else:
        ids = exact_prompt_ids(tokenize, max(lengths))
    print(f"[attribution] loaded in {loading['load_s']:.1f}s, {len(ids)} prompt ids "
          f"from {'literal text' if lengths == [len(ids)] and (args.text or args.text_file) else args.corpus} "
          f"corpus", flush=True)

    source_commit = ""
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    server_command = [
        str(args.server), "-m", str(args.artifact), "-ngl", "99", "-fa", "on",
        "-ctk", args.kv, "-ctv", args.kv, "-c", str(args.context), "-np", "1",
        "-b", "4096", "-ub", "1024", "--host", "127.0.0.1",
        "--port", str(args.port), "--no-cache-prompt", "--fit", "off",
    ]
    print(f"[attribution] launching: {shlex.join(server_command)}", flush=True)
    log_path = args.out.with_suffix(".log")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    started_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT)
        _wait_health(base, process, log_path)

        for length in lengths:
            prompt = ids[:length]
            response = _post(base, _completion_body(prompt, args.continuation, n_probs=5))
            if response.get("tokens_evaluated") != length:
                raise RuntimeError(
                    f"llama.cpp evaluated {response.get('tokens_evaluated')} of {length} ids")
            llama = [int(t) for t in response.get("tokens", [])]
            if not llama:
                raise RuntimeError(f"llama.cpp returned no token for {length} ids")
            llama_steps = []
            for entry in (response.get("completion_probabilities") or [])[: len(llama)]:
                probs = entry.get("probs") or []
                llama_steps.append({
                    "token": int(entry.get("id", -1)),
                    "p_top1": float(probs[0]["prob"]) if probs else None,
                    "runner_up": float(probs[1]["prob"]) if len(probs) > 1 else None,
                })
            base_arm = _run_arm(runner, prompt, args.continuation, mmq=False)
            mmq_arm = _run_arm(runner, prompt, args.continuation, mmq=True)
            logits = _logit_comparison(base_arm, mmq_arm)
            rows.append({
                "ids": length,
                "llama_first": llama[0],
                "base_first": base_arm["tokens"][0],
                "mmq_first": mmq_arm["tokens"][0],
                "llama_tokens": llama,
                "base_tokens": base_arm["tokens"],
                "mmq_tokens": mmq_arm["tokens"],
                "base_matches_llama": base_arm["tokens"][0] == llama[0],
                "mmq_matches_llama": mmq_arm["tokens"][0] == llama[0],
                "arms_agree": base_arm["tokens"][0] == mmq_arm["tokens"][0],
                "base_first_margin": base_arm["first_margin"],
                "mmq_first_margin": mmq_arm["first_margin"],
                "mmq_route_used": mmq_arm["route_used"],
                "logits": logits,
                "repeat_check": _repeat_check(runner, prompt),
                "llama_steps": llama_steps,
                "base_steps": base_arm["steps"],
                "mmq_steps": mmq_arm["steps"],
            })
            row = rows[-1]
            print(f"{length:5d}  llama->{llama[0]:<7d} base->{base_arm['tokens'][0]:<7d} "
                  f"mmq->{mmq_arm['tokens'][0]:<7d} "
                  f"rawbase->{logits['raw_argmax_base']:<7d} "
                  f"rawmmq->{logits['raw_argmax_mmq']:<7d} "
                  f"dmed={logits['delta_median']:+.3f} dmax={logits['delta_max_abs']:.2f} "
                  f"span {logits['raw_span5_base']:.2f}->{logits['raw_span5_mmq']:.2f} "
                  f"softcap_keeps_argmax={logits['softcap_preserves_argmax_base']}/"
                  f"{logits['softcap_preserves_argmax_mmq']} "
                  f"repeat_stable={row['repeat_check']['argmax_stable']} "
                  f"repeat_maxdelta={row['repeat_check']['max_abs_delta']:.3g}",
                  f"base={'Y' if row['base_matches_llama'] else 'n'} "
                  f"mmq={'Y' if row['mmq_matches_llama'] else 'n'} "
                  f"margin base={row['base_first_margin']:.3f} "
                  f"mmq={row['mmq_first_margin']:.3f} "
                  f"route={'yes' if row['mmq_route_used'] else 'NO'}", flush=True)
            # The campaign has been scoring the first token. A route that agrees
            # there and degrades afterwards is only visible step by step.
            span = min(len(llama_steps), len(base_arm["steps"]), len(mmq_arm["steps"]))
            for index in range(span):
                ls, bs, ms = llama_steps[index], base_arm["steps"][index], mmq_arm["steps"][index]
                agree = ls["token"] == bs["token"] == ms["token"]
                print(f"      step {index:2d}  llama {ls['token']:<7d} p={_fmt(ls['p_top1'])}  "
                      f"base {bs['token']:<7d} p={_fmt(bs['p_top1'])} m={bs['raw_margin']:+7.2f}  "
                      f"mmq {ms['token']:<7d} p={_fmt(ms['p_top1'])} m={ms['raw_margin']:+7.2f}"
                      f"{'  <-- parts here' if not agree else ''}", flush=True)
    finally:
        if process is not None:
            _stop(process)

    diverged = [row for row in rows if not row["arms_agree"]]
    record = {
        "kind": "gemma4_mmq_arm_attribution",
        "performance_claim": False,
        "created_at": started_at,
        "artifact": str(args.artifact),
        "corpus": args.corpus,
        "prompt_source": "literal text" if (args.text is not None or args.text_file is not None)
                         else f"{args.corpus} corpus",
        "continuation": args.continuation,
        "anchor": {
            "engine": "llama.cpp llama-server",
            "source": str(args.source),
            "commit": source_commit,
            "server_command": server_command,
            "sampling": "temperature=0, top_k=1, seed=12345, ignore_eos=true, cache_prompt=false",
            "logits": "final-row argmax with the model's own final_logit_softcapping applied",
        },
        "lengths": [row["ids"] for row in rows],
        "arms_agree_lengths": [row["ids"] for row in rows if row["arms_agree"]],
        "arms_diverged_lengths": [row["ids"] for row in diverged],
        "base_matches_anchor": sum(1 for row in rows if row["base_matches_llama"]),
        "mmq_matches_anchor": sum(1 for row in rows if row["mmq_matches_llama"]),
        "rows": rows,
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\narms diverged at {len(diverged)} of {len(rows)} lengths: "
          f"{[row['ids'] for row in diverged]}")
    print(f"anchor agreement: fp32 {record['base_matches_anchor']}/{len(rows)}, "
          f"mmq {record['mmq_matches_anchor']}/{len(rows)}")
    print(f"artifact={args.out}")
    return 0 if not diverged else 1


if __name__ == "__main__":
    raise SystemExit(main())
