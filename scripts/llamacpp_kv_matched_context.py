#!/usr/bin/env python3
# ruff: noqa: E402
"""Build and run the exact llama.cpp F16-vs-Q8_0 KV matched-context gate.

The companion C++ harness uses llama.cpp's public C API, captures only the
prompt-final and requested decode logits, and forces F16-reference tokens into
the Q8_0 candidate. This wrapper adds reproducible build commands, binary/model
fingerprints, source-tree state, and structural validation to the JSON result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import _file_fingerprint

DEFAULT_LLAMA_SOURCE = Path("/home/lhl/llama.cpp/llama.cpp-hip")
DEFAULT_LLAMA_BUILD = DEFAULT_LLAMA_SOURCE / "build-gfx1100-therock715"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_BUILD_ROOT = Path("/tmp/hipengine-llamacpp-kv-matched")
CPP_SOURCE = REPO_ROOT / "scripts" / "llamacpp_kv_matched_context.cpp"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _build_key(*, source: Path, llama_include: Path, llama_library: Path) -> str:
    digest = hashlib.sha256()
    for path in (source, llama_include, llama_library):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _build_command(
    *,
    compiler: str,
    source: Path,
    llama_source: Path,
    llama_build: Path,
    output: Path,
) -> list[str]:
    bin_dir = llama_build / "bin"
    return [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source),
        f"-I{llama_source / 'include'}",
        f"-I{llama_source / 'ggml' / 'include'}",
        f"-L{bin_dir}",
        f"-Wl,-rpath,{bin_dir}",
        "-Wl,--allow-shlib-undefined",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-ldl",
        "-pthread",
        "-o",
        str(output),
    ]


def _read_prompt_tokens(path: Path) -> list[int]:
    try:
        fields = path.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise ValueError(f"failed to read prompt token file {path}: {exc}") from exc
    if not fields:
        raise ValueError(f"prompt token file is empty: {path}")
    tokens: list[int] = []
    for index, field in enumerate(fields):
        try:
            token = int(field, 10)
        except ValueError as exc:
            raise ValueError(f"invalid prompt token at index {index}: {field!r}") from exc
        if not 0 <= token <= (1 << 31) - 1:
            raise ValueError(f"prompt token at index {index} is outside signed int32: {token}")
        tokens.append(token)
    return tokens


def _token_ids_sha256(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(4, byteorder="little", signed=True))
    return digest.hexdigest()


def _prompt_metadata(args: argparse.Namespace) -> dict[str, Any]:
    token_file = getattr(args, "prompt_token_file", None)
    if token_file is None:
        tokens = [int(args.prompt_token_id)] * int(args.prompt_length)
        return {
            "mode": "repeated_token",
            "token_id": int(args.prompt_token_id),
            "token_count": len(tokens),
            "distinct_tokens": 1,
            "prefix_token_ids_sample": tokens[:16],
            "token_ids_int32_le_sha256": _token_ids_sha256(tokens),
        }
    tokens = _read_prompt_tokens(token_file)
    if len(tokens) != int(args.prompt_length):
        raise ValueError(
            f"prompt token file contains {len(tokens)} tokens, expected exactly {args.prompt_length}"
        )
    return {
        "mode": "token_file",
        "token_file": str(token_file),
        "token_count": len(tokens),
        "distinct_tokens": len(set(tokens)),
        "prefix_token_ids_sample": tokens[:16],
        "token_ids_int32_le_sha256": _token_ids_sha256(tokens),
        "token_file_fingerprint": _file_fingerprint(token_file),
    }


def _run_command(args: argparse.Namespace, *, binary: Path, cpp_json: Path) -> list[str]:
    command = [
        str(binary),
        "--model",
        str(args.model),
        "--json",
        str(cpp_json),
        "--prompt-token-id",
        str(args.prompt_token_id),
        "--prompt-length",
        str(args.prompt_length),
        "--decode-steps",
        str(args.decode_steps),
        "--ctx-size",
        str(args.ctx_size or args.prompt_length + args.decode_steps + 1),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--threads",
        str(args.threads),
        "--reference-cache",
        args.reference_cache,
        "--candidate-cache",
        args.candidate_cache,
        "--flash-attn",
        "on" if args.flash_attn else "off",
        "--kl-threshold",
        str(args.kl_threshold),
        "--top1-threshold",
        str(args.top1_threshold),
    ]
    prompt_token_file = getattr(args, "prompt_token_file", None)
    if prompt_token_file is not None:
        command.extend(("--prompt-token-file", str(prompt_token_file)))
    candidate_cache_k = getattr(args, "candidate_cache_k", None)
    candidate_cache_v = getattr(args, "candidate_cache_v", None)
    if candidate_cache_k is not None:
        command.extend(("--candidate-cache-k", str(candidate_cache_k)))
    if candidate_cache_v is not None:
        command.extend(("--candidate-cache-v", str(candidate_cache_v)))
    reference_logits_bin = getattr(args, "reference_logits_bin", None)
    if reference_logits_bin is not None:
        command.extend(("--reference-logits-bin", str(reference_logits_bin)))
    return command


def _validate_cpp_payload(
    payload: dict[str, Any],
    *,
    prompt_length: int,
    decode_steps: int,
    prompt_mode: str | None = None,
) -> None:
    if payload.get("mode") != "llamacpp_kv_matched_context":
        raise ValueError("unexpected C++ harness mode")
    if payload.get("prompt_length") != prompt_length or payload.get("decode_steps") != decode_steps:
        raise ValueError("C++ harness workload does not match request")
    prompt = payload.get("prompt", {})
    if prompt.get("token_count") != prompt_length:
        raise ValueError("C++ harness prompt token count mismatch")
    if prompt_mode is not None and prompt.get("mode") != prompt_mode:
        raise ValueError("C++ harness prompt mode does not match request")
    positions = decode_steps + 1
    if payload.get("positions") != positions:
        raise ValueError("C++ harness position count mismatch")
    reference = payload.get("reference", {})
    candidate = payload.get("candidate", {})
    comparison = payload.get("matched_context", {})
    reference_top1 = list(reference.get("top1_ids", ()))
    candidate_top1 = list(candidate.get("top1_ids", ()))
    forced_inputs = list(candidate.get("decode_input_ids", ()))
    if len(reference_top1) != positions or len(candidate_top1) != positions:
        raise ValueError("C++ harness top-1 row count mismatch")
    if forced_inputs != reference_top1[:decode_steps]:
        raise ValueError("candidate did not consume the F16 reference token history")
    kls = [float(item) for item in comparison.get("kl", ())]
    matches = [bool(item) for item in comparison.get("top1_matches", ())]
    if len(kls) != positions or len(matches) != positions:
        raise ValueError("C++ harness metric row count mismatch")
    mean_kl = sum(kls) / len(kls)
    max_kl = max(kls)
    top1 = sum(matches) / len(matches)
    if abs(mean_kl - float(comparison["mean_kl"])) > 1.0e-12:
        raise ValueError("mean KL does not match per-position rows")
    if abs(max_kl - float(comparison["max_kl"])) > 1.0e-12:
        raise ValueError("max KL does not match per-position rows")
    if abs(top1 - float(comparison["top1_agreement"])) > 1.0e-12:
        raise ValueError("top-1 agreement does not match per-position rows")
    if not reference.get("finite_logits") or not candidate.get("finite_logits"):
        raise ValueError("non-finite logits in C++ harness result")


def _version_text(binary: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return (completed.stdout + completed.stderr).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _prompt_metadata(args)
    llama_bin = args.llama_build / "bin"
    llama_library = llama_bin / "libllama.so.0.0.9648"
    if not llama_library.exists():
        candidates = sorted(llama_bin.glob("libllama.so.*.*.*"))
        if not candidates:
            raise FileNotFoundError(f"no versioned libllama found under {llama_bin}")
        llama_library = candidates[-1]
    key = _build_key(
        source=CPP_SOURCE,
        llama_include=args.llama_source / "include" / "llama.h",
        llama_library=llama_library,
    )
    build_dir = args.build_root / key
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "llamacpp-kv-matched-context"
    build_command = _build_command(
        compiler=args.compiler,
        source=CPP_SOURCE,
        llama_source=args.llama_source,
        llama_build=args.llama_build,
        output=binary,
    )
    if args.rebuild or not binary.exists():
        subprocess.run(build_command, check=True)
    cpp_json = build_dir / "result.json"
    if args.reference_logits_bin is not None:
        args.reference_logits_bin.parent.mkdir(parents=True, exist_ok=True)
    run_command = _run_command(args, binary=binary, cpp_json=cpp_json)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            (
                str(Path("/opt/rocm/lib")),
                str(llama_bin),
                env.get("LD_LIBRARY_PATH", ""),
            ),
        )
    )
    subprocess.run(run_command, env=env, check=True)
    cpp_payload = json.loads(cpp_json.read_text(encoding="utf-8"))
    _validate_cpp_payload(
        cpp_payload,
        prompt_length=args.prompt_length,
        decode_steps=args.decode_steps,
        prompt_mode=str(prompt["mode"]),
    )
    libraries = {}
    for name in ("libllama.so.0", "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0", "libggml-hip.so.0"):
        path = (llama_bin / name).resolve()
        libraries[name] = {"path": str(path), "sha256": _sha256_file(path)}
    host_state = _git_state(REPO_ROOT)
    llama_state = _git_state(args.llama_source)
    reference_logits_dump = None
    if args.reference_logits_bin is not None:
        reference_logits_dump = {
            "path": str(args.reference_logits_bin),
            "size_bytes": args.reference_logits_bin.stat().st_size,
            "sha256": _sha256_file(args.reference_logits_bin),
        }
    return {
        "schema": 1,
        "status": cpp_payload["status"],
        "mode": "llamacpp_kv_matched_context_driver",
        "performance_claim": False,
        "command": shlex.join(["python3", "scripts/llamacpp_kv_matched_context.py", *sys.argv[1:]]),
        "build": {
            "key": key,
            "compiler": args.compiler,
            "command": shlex.join(build_command),
            "binary": str(binary),
            "binary_sha256": _sha256_file(binary),
            "cpp_source": str(CPP_SOURCE.relative_to(REPO_ROOT)),
            "cpp_source_sha256": _sha256_file(CPP_SOURCE),
        },
        "reference_logits_dump": reference_logits_dump,
        "prompt": prompt,
        "provenance": {
            "hipengine": host_state,
            "llama_cpp": llama_state,
            "llama_build_dir": str(args.llama_build),
            "llama_completion_version": _version_text(llama_bin / "llama-completion", env),
            "libraries": libraries,
            "model_fingerprint": _file_fingerprint(args.model),
            "environment": {
                "HIP_VISIBLE_DEVICES": env.get("HIP_VISIBLE_DEVICES"),
                "ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"),
                "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"),
            },
        },
        "result": cpp_payload,
        "caveats": [
            "F16 and Q8_0 refer only to llama.cpp K/V cache storage; reference and candidate use identical Q4_K_M model weights.",
            "The external llama.cpp source tree may contain local instrumentation; exact shared-library hashes identify the measured build.",
            "Candidate decode inputs are structurally validated against the F16 reference top-1 history.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, default=DEFAULT_LLAMA_SOURCE)
    parser.add_argument("--llama-build", type=Path, default=DEFAULT_LLAMA_BUILD)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument(
        "--prompt-token-file",
        type=Path,
        help="Whitespace-delimited token IDs; must contain exactly --prompt-length entries",
    )
    parser.add_argument("--prompt-length", type=int, default=131072)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--ctx-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--reference-cache", choices=("f16", "q8_0"), default="f16")
    parser.add_argument("--candidate-cache", choices=("f16", "q8_0"), default="q8_0")
    parser.add_argument("--candidate-cache-k", choices=("f16", "q8_0"))
    parser.add_argument("--candidate-cache-v", choices=("f16", "q8_0"))
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--reference-logits-bin", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.prompt_token_id < 0 or args.prompt_length <= 0 or args.decode_steps < 0:
        raise ValueError("prompt token ID/length and decode steps are invalid")
    if args.batch_size <= 0 or args.ubatch_size <= 0 or args.ubatch_size > args.batch_size:
        raise ValueError("batch/ubatch sizes are invalid")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
