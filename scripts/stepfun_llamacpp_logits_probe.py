#!/usr/bin/env python3
"""Capture same-prompt llama.cpp logits for the StepFun oracle blocker.

The probe uses the retained host prompt-smoke artifact as the source of truth for
prompt text and token IDs, runs the built llama-debug entrypoint with
--save-logits, and retains a compact JSON summary with logits/ranks for the host
expected token and the llama.cpp-generated mismatch token. Raw llama-debug output
files are intermediate evidence; the committed artifact is the compact JSON.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_llamacpp_logits_preflight as preflight_mod

DEFAULT_OUTPUT = Path("benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json")
DEFAULT_PROMPT_ARTIFACT = status_mod.DEFAULT_PROMPT_ARTIFACT
DEFAULT_LLAMA_DEBUG = preflight_mod.DEFAULT_LLAMA_CLI
DEFAULT_MODEL = Path("/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf")
DEFAULT_RAW_OUTPUT_DIR = Path("/tmp/hipengine-stepfun-q3kl-llamacpp-logits-probe")
DEFAULT_ARTIFACT_DATE = "2026-06-15"
DEFAULT_GENERATED_TOKEN_ID = 671
DEFAULT_GENERATED_TOKEN_TEXT = "The"
SAME_PROMPT_SPECIAL_HELP_MARKERS = ("--special", "--parse-special", "--token-ids")
PROMPT_TOKEN_SOURCE_TEXT = "text"
PROMPT_TOKEN_SOURCE_RETAINED_IDS = "retained-input-ids"
PROMPT_TOKEN_SOURCE_CHOICES = (PROMPT_TOKEN_SOURCE_TEXT, PROMPT_TOKEN_SOURCE_RETAINED_IDS)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-artifact", type=Path, default=DEFAULT_PROMPT_ARTIFACT)
    parser.add_argument("--llama-debug", type=Path, default=DEFAULT_LLAMA_DEBUG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generated-token-id", type=int, default=DEFAULT_GENERATED_TOKEN_ID)
    parser.add_argument("--generated-token-text", default=DEFAULT_GENERATED_TOKEN_TEXT)
    parser.add_argument(
        "--prompt-token-source",
        choices=PROMPT_TOKEN_SOURCE_CHOICES,
        default=PROMPT_TOKEN_SOURCE_TEXT,
        help=(
            "How the helper should receive the retained prompt. 'text' preserves the current "
            "llama-debug prompt path; 'retained-input-ids' passes the prompt artifact's token IDs "
            "via --token-ids for a future same-prompt helper."
        ),
    )
    parser.add_argument("--artifact-date", default=DEFAULT_ARTIFACT_DATE)
    parser.add_argument("--execute", action="store_true", help="Run llama-debug and parse saved logits.")
    parser.add_argument(
        "--force-execute-without-special",
        action="store_true",
        help="Attempt execution even if the probe binary help lacks a special-token parsing flag.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw llama-debug .bin/.txt outputs after parsing. Default removes them.",
    )
    parser.add_argument(
        "--llama-arg",
        action="append",
        default=None,
        help="Additional argument appended to llama-debug; repeat for each token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output atomically to this path instead of stdout.",
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical StepFun artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument("--same-prompt-tokens-match-only", action="store_true")
    parser.add_argument("--expected-outranks-generated-only", action="store_true")
    parser.add_argument("--sha-only", action="store_true")
    parser.add_argument(
        "--verify-probe",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted llama.cpp logits probe artifact with current "
            f"prompt/helper/execution metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-probe, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-probe, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-probe, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _write_text_atomic(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _emit_json(payload: object, *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    _write_text_atomic(output, text)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "artifact_kind": None, "status": None}
    payload = _load_json_object(path)
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _marker_in_help(text: str, marker: str) -> bool:
    import re

    return re.search(rf"(?<!\S){re.escape(marker)}(?![\w-])", text) is not None


def _llama_help_capability(binary: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "path": str(binary),
        "exists": binary.exists(),
        "executable": os.access(binary, os.X_OK),
        "help_status": "not_run",
        "help_returncode": None,
        "same_prompt_special_markers_checked": list(SAME_PROMPT_SPECIAL_HELP_MARKERS),
        "matched_same_prompt_special_markers": [],
        "same_prompt_special_token_flag_present": False,
    }
    if not info["exists"] or not info["executable"]:
        info["help_status"] = "unavailable"
        return info
    try:
        completed = subprocess.run(
            [str(binary), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        info["help_status"] = f"error:{type(exc).__name__}"
        info["help_error"] = str(exc)
        return info
    text = (completed.stdout or "") + (completed.stderr or "")
    matched = [marker for marker in SAME_PROMPT_SPECIAL_HELP_MARKERS if _marker_in_help(text, marker)]
    parse_special_present = _marker_in_help(text, "--parse-special")
    token_ids_present = _marker_in_help(text, "--token-ids")
    output_special_present = _marker_in_help(text, "--special")
    info.update(
        {
            "help_status": "executed",
            "help_returncode": completed.returncode,
            "matched_same_prompt_special_markers": matched,
            "same_prompt_special_token_flag_present": bool(matched),
            "parse_special_flag_present": parse_special_present,
            "token_ids_flag_present": token_ids_present,
            "output_special_flag_present": output_special_present,
            "same_prompt_text_tokenization_capable": parse_special_present,
            "same_prompt_retained_token_ids_capable": token_ids_present,
        }
    )
    return info


def _llama_version(binary: Path) -> str | None:
    if not binary.exists():
        return None
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {type(exc).__name__}: {exc}"
    return (completed.stdout + completed.stderr).strip() or None


def _model_stem(model: Path) -> str:
    return model.stem


def _raw_output_paths(model: Path, raw_output_dir: Path) -> dict[str, Path]:
    base = raw_output_dir / f"llamacpp-{_model_stem(model)}"
    return {
        "logits_bin": Path(str(base) + ".bin"),
        "logits_txt": Path(str(base) + ".txt"),
        "prompt_txt": Path(str(base) + "-prompt.txt"),
        "tokens_bin": Path(str(base) + "-tokens.bin"),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_output_refs(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    refs: dict[str, dict[str, object]] = {}
    for key, path in paths.items():
        refs[key] = {
            "path": str(path),
            "exists": path.exists(),
            "nbytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    return refs


def _cleanup_raw_outputs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _retained_token_ids_csv(prompt_payload: dict[str, object]) -> str:
    input_ids = prompt_payload.get("input_ids")
    if not isinstance(input_ids, list) or not all(isinstance(item, int) for item in input_ids):
        raise ValueError("prompt artifact missing integer input_ids for retained-input-ids mode")
    return ",".join(str(item) for item in input_ids)


def _build_command(
    *,
    llama_debug: Path,
    model: Path,
    prompt: str,
    raw_output_dir: Path,
    prompt_token_source: str = PROMPT_TOKEN_SOURCE_TEXT,
    retained_token_ids_csv: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    prompt_args = ["--prompt", prompt]
    if prompt_token_source == PROMPT_TOKEN_SOURCE_RETAINED_IDS:
        if not retained_token_ids_csv:
            raise ValueError("retained-input-ids mode requires retained_token_ids_csv")
        prompt_args.extend(["--token-ids", retained_token_ids_csv])
    elif prompt_token_source != PROMPT_TOKEN_SOURCE_TEXT:
        raise ValueError(f"unsupported prompt token source: {prompt_token_source}")
    return [
        str(llama_debug),
        "--model",
        str(model),
        *prompt_args,
        "--save-logits",
        "--logits-output-dir",
        str(raw_output_dir),
        "--gpu-layers",
        "999",
        "--no-warmup",
        "--no-perf",
        "--special",
        "--override-kv",
        "tokenizer.ggml.add_bos_token=bool:false",
        *extra_args,
    ]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> tuple[str, str, dict[str, object]]:
    termination: dict[str, object] = {
        "termination_method": "os.killpg",
        "termination_signal": "SIGKILL",
        "process_group_started": True,
    }
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:  # pragma: no cover - process exited during timeout handling
        termination["process_exited_before_signal"] = True
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive fallback
        termination["fallback_proc_kill_used"] = True
        proc.kill()
        stdout, stderr = proc.communicate()
    return _as_text(stdout), _as_text(stderr), termination


def _run_with_timeout(command: Sequence[str], timeout_s: float) -> tuple[str, int | None, str, str, float, dict[str, object] | None]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return "executed", proc.returncode, _as_text(stdout), _as_text(stderr), time.perf_counter() - started, None
    except subprocess.TimeoutExpired as exc:
        stdout_text, stderr_text, termination = _terminate_process_group(proc)
        termination.update({"timeout_s": timeout_s, "timeout_reached": True})
        return (
            "timeout",
            None,
            _as_text(exc.stdout) + stdout_text,
            _as_text(exc.stderr) + stderr_text,
            time.perf_counter() - started,
            termination,
        )


def _read_logits(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"logits binary size is not a multiple of float32: {path}")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":  # pragma: no cover - current target is little endian
        values.byteswap()
    return list(values)


def _parse_prompt_tokens(prompt_txt: Path) -> dict[str, object]:
    text = prompt_txt.read_text(errors="replace")
    n_tokens: int | None = None
    token_ids: list[int] | None = None
    prompt: str | None = None
    for line in text.splitlines():
        if line.startswith("prompt: "):
            prompt = line[len("prompt: ") :]
        elif line.startswith("n_tokens: "):
            n_tokens = int(line[len("n_tokens: ") :])
        elif line.startswith("token ids: "):
            rest = line[len("token ids: ") :].strip()
            token_ids = [] if not rest else [int(part.strip()) for part in rest.split(",")]
    return {"prompt": prompt, "n_tokens": n_tokens, "token_ids": token_ids}


def _rank(logits: Sequence[float], token_id: int) -> int:
    value = logits[token_id]
    return 1 + sum(1 for item in logits if item > value)


def _top_tokens(logits: Sequence[float], top_k: int) -> list[dict[str, object]]:
    pairs = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {"rank": rank, "token_id": token_id, "logit": logit}
        for rank, (token_id, logit) in enumerate(pairs, start=1)
    ]


def _token_record(
    *,
    logits: Sequence[float],
    token_id: int,
    token_text: object,
    host_logit: object = None,
    host_rank: object = None,
) -> dict[str, object]:
    return {
        "token_id": token_id,
        "token_text": token_text,
        "llamacpp_logit": logits[token_id],
        "llamacpp_rank": _rank(logits, token_id),
        "host_logit": host_logit,
        "host_rank": host_rank,
    }


def _build_planned_report(
    *,
    prompt_artifact: Path,
    prompt_payload: dict[str, object],
    llama_debug: Path,
    model: Path,
    raw_output_dir: Path,
    command: list[str],
    generated_token_id: int,
    generated_token_text: str,
    artifact_date: str,
    top_k: int,
    timeout_s: float,
    prompt_token_source: str,
    retained_token_ids_csv: str | None,
) -> dict[str, object]:
    raw_paths = _raw_output_paths(model, raw_output_dir)
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_probe",
        "date": artifact_date,
        "status": "planned",
        "ready": False,
        "prompt_artifact": _artifact_ref(prompt_artifact),
        "llama_debug": str(llama_debug),
        "llama_cpp_version": _llama_version(llama_debug),
        "llama_debug_same_prompt_capability": _llama_help_capability(llama_debug),
        "model": str(model),
        "raw_output_dir": str(raw_output_dir),
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "raw_outputs_retained_after_parse": False,
        "timeout_s": timeout_s,
        "top_k": top_k,
        "command": command,
        "command_shell": shlex.join(command),
        "prompt": prompt_payload.get("prompt"),
        "prompt_length": prompt_payload.get("prompt_length"),
        "input_ids": prompt_payload.get("input_ids"),
        "prompt_token_source": prompt_token_source,
        "retained_token_ids_argument": retained_token_ids_csv,
        "retained_token_ids_argument_present": bool(retained_token_ids_csv),
        "target": {
            "canonical_backend": "vulkan",
            "readiness_gate": "oracle_parity",
            "unresolved_evidence_gap": "generated_text_matches_target",
            "expected_next_token_id": prompt_payload.get("next_token_id"),
            "expected_next_token_text": prompt_payload.get("next_token_text"),
            "generated_first_token_id": generated_token_id,
            "generated_first_token_text": generated_token_text,
        },
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": "This artifact captures same-prompt logits evidence only; generated_text_matches_target remains unresolved.",
        },
    }


def build_llamacpp_logits_probe(
    *,
    prompt_artifact: Path = DEFAULT_PROMPT_ARTIFACT,
    llama_debug: Path = DEFAULT_LLAMA_DEBUG,
    model: Path = DEFAULT_MODEL,
    raw_output_dir: Path = DEFAULT_RAW_OUTPUT_DIR,
    timeout_s: float = 900.0,
    top_k: int = 10,
    generated_token_id: int = DEFAULT_GENERATED_TOKEN_ID,
    generated_token_text: str = DEFAULT_GENERATED_TOKEN_TEXT,
    prompt_token_source: str = PROMPT_TOKEN_SOURCE_TEXT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
    execute: bool = False,
    force_execute_without_special: bool = False,
    keep_raw: bool = False,
    extra_llama_args: Sequence[str] = (),
) -> dict[str, object]:
    prompt_payload = _load_json_object(prompt_artifact)
    prompt = prompt_payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"prompt artifact missing string prompt: {prompt_artifact}")
    retained_token_ids_csv = (
        _retained_token_ids_csv(prompt_payload)
        if prompt_token_source == PROMPT_TOKEN_SOURCE_RETAINED_IDS
        else None
    )
    command = _build_command(
        llama_debug=llama_debug,
        model=model,
        prompt=prompt,
        raw_output_dir=raw_output_dir,
        prompt_token_source=prompt_token_source,
        retained_token_ids_csv=retained_token_ids_csv,
        extra_args=extra_llama_args,
    )
    report = _build_planned_report(
        prompt_artifact=prompt_artifact,
        prompt_payload=prompt_payload,
        llama_debug=llama_debug,
        model=model,
        raw_output_dir=raw_output_dir,
        command=command,
        generated_token_id=generated_token_id,
        generated_token_text=generated_token_text,
        artifact_date=artifact_date,
        top_k=top_k,
        timeout_s=timeout_s,
        prompt_token_source=prompt_token_source,
        retained_token_ids_csv=retained_token_ids_csv,
    )
    if not execute:
        return report
    capability = report.get("llama_debug_same_prompt_capability")
    if prompt_token_source == PROMPT_TOKEN_SOURCE_RETAINED_IDS:
        special_ready = (
            isinstance(capability, dict)
            and capability.get("same_prompt_retained_token_ids_capable") is True
        )
        missing_evidence = "llama_debug_retained_token_ids_input_present"
        blocked_reason = (
            "built llama-debug exposes --save-logits but does not accept retained token IDs, "
            "so it cannot bypass text tokenization for the exact StepFun prompt IDs"
        )
    else:
        special_ready = (
            isinstance(capability, dict)
            and capability.get("same_prompt_special_token_flag_present") is True
        )
        missing_evidence = "llama_debug_same_prompt_special_token_support_present"
        blocked_reason = (
            "built llama-debug exposes --save-logits but does not accept a special-token "
            "parsing flag, so it cannot reproduce the retained StepFun prompt token IDs"
        )
    if not special_ready and not force_execute_without_special:
        report.update(
            {
                "status": "blocked",
                "ready": False,
                "missing_evidence": [missing_evidence],
                "blocked_reason": blocked_reason,
                "next_action": (
                    "add or build a logits-dump helper that tokenizes the retained prompt with "
                    "special tokens enabled or accepts explicit token IDs, then rerun this probe"
                ),
            }
        )
        return report

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = _raw_output_paths(model, raw_output_dir)
    _cleanup_raw_outputs(raw_paths)
    status, returncode, stdout, stderr, elapsed_s, timeout_termination = _run_with_timeout(
        command, timeout_s
    )
    report.update(
        {
            "execution_status": status,
            "returncode": returncode,
            "elapsed_s": elapsed_s,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "timeout_termination": timeout_termination,
            "raw_output_refs_before_cleanup": _raw_output_refs(raw_paths),
        }
    )
    if status != "executed" or returncode != 0:
        report.update(
            {
                "status": status if status == "timeout" else "failed",
                "ready": False,
                "blocked_reason": "llama-debug execution did not complete successfully",
            }
        )
        if not keep_raw:
            _cleanup_raw_outputs(raw_paths)
        return report

    missing_raw = [key for key, path in raw_paths.items() if not path.exists()]
    if missing_raw:
        report.update(
            {
                "status": "failed",
                "ready": False,
                "missing_raw_outputs": missing_raw,
                "blocked_reason": "llama-debug completed but did not produce all expected logits outputs",
            }
        )
        if not keep_raw:
            _cleanup_raw_outputs(raw_paths)
        return report

    logits = _read_logits(raw_paths["logits_bin"])
    prompt_tokens = _parse_prompt_tokens(raw_paths["prompt_txt"])
    input_ids = prompt_payload.get("input_ids") if isinstance(prompt_payload.get("input_ids"), list) else []
    same_prompt_tokens_match = prompt_tokens.get("token_ids") == input_ids
    expected_token_id = int(prompt_payload.get("next_token_id"))
    expected_token = _token_record(
        logits=logits,
        token_id=expected_token_id,
        token_text=prompt_payload.get("next_token_text"),
        host_logit=prompt_payload.get("next_token_logit"),
        host_rank=1,
    )
    generated_token = _token_record(
        logits=logits,
        token_id=generated_token_id,
        token_text=generated_token_text,
    )
    top = _top_tokens(logits, top_k)
    comparison = {
        "expected_token_id": expected_token_id,
        "generated_token_id": generated_token_id,
        "llamacpp_top1_token_id": top[0]["token_id"] if top else None,
        "llamacpp_expected_outranks_generated": expected_token["llamacpp_rank"] < generated_token["llamacpp_rank"],
        "llamacpp_expected_is_top1": expected_token["llamacpp_rank"] == 1,
        "llamacpp_generated_is_top1": generated_token["llamacpp_rank"] == 1,
        "host_expected_is_top1": True,
    }
    report.update(
        {
            "status": "captured" if same_prompt_tokens_match else "failed",
            "ready": same_prompt_tokens_match,
            "vocab_size": len(logits),
            "same_prompt_tokens_match": same_prompt_tokens_match,
            "prompt_tokens_from_llamacpp": prompt_tokens,
            "expected_token": expected_token,
            "generated_token": generated_token,
            "llamacpp_top_tokens": top,
            "comparison": comparison,
            "blocked_reason": None if same_prompt_tokens_match else "llama-debug tokenization did not match retained host prompt input IDs",
        }
    )
    if not keep_raw:
        _cleanup_raw_outputs(raw_paths)
        report["raw_outputs_removed_after_parse"] = True
        report["raw_output_refs_after_cleanup"] = _raw_output_refs(raw_paths)
    else:
        report["raw_outputs_removed_after_parse"] = False
        report["raw_output_refs_after_cleanup"] = _raw_output_refs(raw_paths)
    return report


def verify_llamacpp_logits_probe(
    probe_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted llama.cpp logits probe artifact with current metadata."""

    persisted = _load_json_object(probe_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "llamacpp_logits_probe_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted llama.cpp logits probe artifact differs from current "
                    "prompt/helper/execution metadata."
                ),
            }
        )
    all_match = not failures
    persisted_capability = persisted.get("llama_debug_same_prompt_capability")
    current_capability = current_report.get("llama_debug_same_prompt_capability")
    persisted_comparison = persisted.get("comparison")
    current_comparison = current_report.get("comparison")
    return {
        "schema_version": 1,
        "artifact_path": str(probe_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_ready": persisted.get("ready"),
        "current_ready": current_report.get("ready"),
        "persisted_prompt_token_source": persisted.get("prompt_token_source"),
        "current_prompt_token_source": current_report.get("prompt_token_source"),
        "persisted_retained_token_ids_argument_present": persisted.get(
            "retained_token_ids_argument_present"
        ),
        "current_retained_token_ids_argument_present": current_report.get(
            "retained_token_ids_argument_present"
        ),
        "persisted_retained_token_ids_argument": persisted.get(
            "retained_token_ids_argument"
        ),
        "current_retained_token_ids_argument": current_report.get(
            "retained_token_ids_argument"
        ),
        "persisted_missing_evidence": persisted.get("missing_evidence"),
        "current_missing_evidence": current_report.get("missing_evidence"),
        "persisted_blocked_reason": persisted.get("blocked_reason"),
        "current_blocked_reason": current_report.get("blocked_reason"),
        "persisted_execution_status": persisted.get("execution_status"),
        "current_execution_status": current_report.get("execution_status"),
        "persisted_same_prompt_tokens_match": persisted.get("same_prompt_tokens_match"),
        "current_same_prompt_tokens_match": current_report.get("same_prompt_tokens_match"),
        "persisted_expected_outranks_generated": (
            persisted_comparison.get("llamacpp_expected_outranks_generated")
            if isinstance(persisted_comparison, dict)
            else None
        ),
        "current_expected_outranks_generated": (
            current_comparison.get("llamacpp_expected_outranks_generated")
            if isinstance(current_comparison, dict)
            else None
        ),
        "persisted_retained_token_ids_capable": (
            persisted_capability.get("same_prompt_retained_token_ids_capable")
            if isinstance(persisted_capability, dict)
            else None
        ),
        "current_retained_token_ids_capable": (
            current_capability.get("same_prompt_retained_token_ids_capable")
            if isinstance(current_capability, dict)
            else None
        ),
        "persisted_text_tokenization_capable": (
            persisted_capability.get("same_prompt_text_tokenization_capable")
            if isinstance(persisted_capability, dict)
            else None
        ),
        "current_text_tokenization_capable": (
            current_capability.get("same_prompt_text_tokenization_capable")
            if isinstance(current_capability, dict)
            else None
        ),
        "persisted_prompt_artifact_sha256": (
            persisted.get("prompt_artifact", {}).get("sha256")
            if isinstance(persisted.get("prompt_artifact"), dict)
            else None
        ),
        "current_prompt_artifact_sha256": (
            current_report.get("prompt_artifact", {}).get("sha256")
            if isinstance(current_report.get("prompt_artifact"), dict)
            else None
        ),
        "persisted_no_claim_policy": persisted.get("no_claim_policy"),
        "current_no_claim_policy": current_report.get("no_claim_policy"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_llamacpp_logits_probe(
        prompt_artifact=args.prompt_artifact,
        llama_debug=args.llama_debug,
        model=args.model,
        raw_output_dir=args.raw_output_dir,
        timeout_s=args.timeout_s,
        top_k=args.top_k,
        generated_token_id=args.generated_token_id,
        generated_token_text=args.generated_token_text,
        prompt_token_source=args.prompt_token_source,
        artifact_date=args.artifact_date,
        execute=args.execute,
        force_execute_without_special=args.force_execute_without_special,
        keep_raw=args.keep_raw,
        extra_llama_args=list(args.llama_arg or ()),
    )
    if args.verify_probe is not None:
        verification = verify_llamacpp_logits_probe(
            args.verify_probe,
            current_report=report,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        if args.sha_only:
            payload = status_mod._stable_json_sha256(payload)
        output = DEFAULT_OUTPUT if args.default_output else args.output
        _emit_json(payload, pretty=args.pretty, output=output)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = report["status"]
    elif args.same_prompt_tokens_match_only:
        payload = report.get("same_prompt_tokens_match")
    elif args.expected_outranks_generated_only:
        comparison = report.get("comparison")
        payload = comparison.get("llamacpp_expected_outranks_generated") if isinstance(comparison, dict) else None
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    _emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
