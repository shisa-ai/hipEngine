from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_llamacpp_logits_probe import build_llamacpp_logits_probe, main


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_prompt_artifact(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "partial_prompt_smoke",
            "prompt": "hello",
            "prompt_length": 3,
            "input_ids": [1, 2, 3],
            "next_token_id": 4,
            "next_token_text": " expected",
            "next_token_logit": 9.0,
            "top_tokens": [{"rank": 1, "token_id": 4, "logit": 9.0}],
        },
    )


def _write_fake_llama_debug(
    path: Path,
    *,
    logits: list[float],
    token_ids: list[int],
    help_text: str = "--save-logits --logits-output-dir --special",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import array, pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fake llama-debug')\n"
        "    raise SystemExit(0)\n"
        "if '--help' in sys.argv:\n"
        f"    print({help_text!r})\n"
        "    raise SystemExit(0)\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--logits-output-dir') + 1])\n"
        "model = pathlib.Path(sys.argv[sys.argv.index('--model') + 1])\n"
        "prompt = sys.argv[sys.argv.index('--prompt') + 1]\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "base = out / ('llamacpp-' + model.stem)\n"
        f"vals = {logits!r}\n"
        "arr = array.array('f', vals)\n"
        "arr.tofile(open(str(base) + '.bin', 'wb'))\n"
        "with open(str(base) + '.txt', 'w') as f:\n"
        "    for i, value in enumerate(vals):\n"
        "        f.write(f'{i}: {value}\\n')\n"
        f"token_ids = {token_ids!r}\n"
        "if '--token-ids' in sys.argv:\n"
        "    token_ids = [int(part) for part in sys.argv[sys.argv.index('--token-ids') + 1].split(',') if part]\n"
        "with open(str(base) + '-prompt.txt', 'w') as f:\n"
        "    f.write('prompt: ' + prompt + '\\n')\n"
        "    f.write('n_tokens: ' + str(len(token_ids)) + '\\n')\n"
        "    f.write('token ids: ' + ', '.join(str(x) for x in token_ids) + '\\n')\n"
        "array.array('i', token_ids).tofile(open(str(base) + '-tokens.bin', 'wb'))\n"
    )
    path.chmod(0o755)


def test_stepfun_llamacpp_logits_probe_plans_command(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(fake_debug, logits=[0.0, 1.0, 2.0, 3.0, 4.0], token_ids=[1, 2, 3])
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        generated_token_text="generated",
        execute=False,
        artifact_date="2030-03-04",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_probe"
    assert report["date"] == "2030-03-04"
    assert report["status"] == "planned"
    assert report["ready"] is False
    assert report["prompt_artifact"]["exists"] is True
    assert "--save-logits" in report["command"]
    assert "--logits-output-dir" in report["command"]
    assert "--special" in report["command"]
    assert report["llama_debug_same_prompt_capability"]["same_prompt_special_token_flag_present"] is True
    assert report["llama_debug_same_prompt_capability"]["output_special_flag_present"] is True
    assert report["llama_debug_same_prompt_capability"]["token_ids_flag_present"] is False
    assert report["prompt_token_source"] == "text"
    assert report["retained_token_ids_argument"] is None
    assert report["target"] == {
        "canonical_backend": "vulkan",
        "readiness_gate": "oracle_parity",
        "unresolved_evidence_gap": "generated_text_matches_target",
        "expected_next_token_id": 4,
        "expected_next_token_text": " expected",
        "generated_first_token_id": 2,
        "generated_first_token_text": "generated",
    }


def test_stepfun_llamacpp_logits_probe_plans_retained_token_ids_command(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(
        fake_debug,
        logits=[0.0, 1.0, 2.0, 3.0, 4.0],
        token_ids=[9, 9, 9],
        help_text="--save-logits --logits-output-dir --token-ids",
    )
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        generated_token_text="generated",
        prompt_token_source="retained-input-ids",
        execute=False,
    )

    assert report["prompt_token_source"] == "retained-input-ids"
    assert report["retained_token_ids_argument"] == "1,2,3"
    assert report["retained_token_ids_argument_present"] is True
    assert "--token-ids" in report["command"]
    assert report["command"][report["command"].index("--token-ids") + 1] == "1,2,3"
    assert report["llama_debug_same_prompt_capability"]["same_prompt_retained_token_ids_capable"] is True
    assert report["llama_debug_same_prompt_capability"]["same_prompt_text_tokenization_capable"] is False


def test_stepfun_llamacpp_logits_probe_executes_retained_token_ids_helper(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(
        fake_debug,
        logits=[0.0, 1.5, 8.0, 2.5, 7.0, -1.0],
        token_ids=[9, 9, 9],
        help_text="--save-logits --logits-output-dir --token-ids",
    )
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        generated_token_text="generated",
        top_k=2,
        prompt_token_source="retained-input-ids",
        execute=True,
    )

    assert report["status"] == "captured"
    assert report["ready"] is True
    assert report["same_prompt_tokens_match"] is True
    assert report["prompt_tokens_from_llamacpp"]["token_ids"] == [1, 2, 3]
    assert report["llama_debug_same_prompt_capability"]["same_prompt_retained_token_ids_capable"] is True


def test_stepfun_llamacpp_logits_probe_blocks_retained_token_ids_when_flag_missing(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(
        fake_debug,
        logits=[0.0, 1.0, 2.0, 3.0, 4.0],
        token_ids=[1, 2, 3],
        help_text="--save-logits --logits-output-dir --special",
    )
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        prompt_token_source="retained-input-ids",
        execute=True,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["missing_evidence"] == ["llama_debug_retained_token_ids_input_present"]
    assert report["blocked_reason"] == (
        "built llama-debug exposes --save-logits but does not accept retained token IDs, "
        "so it cannot bypass text tokenization for the exact StepFun prompt IDs"
    )
    assert not (tmp_path / "raw").exists()


def test_stepfun_llamacpp_logits_probe_executes_and_summarizes_logits(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "Step-3.7-flash-Q3_K_L-00001-of-00003.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(
        fake_debug,
        logits=[0.0, 1.5, 8.0, 2.5, 7.0, -1.0],
        token_ids=[1, 2, 3],
    )
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        generated_token_text="generated",
        top_k=3,
        execute=True,
    )

    assert report["status"] == "captured"
    assert report["ready"] is True
    assert report["returncode"] == 0
    assert report["vocab_size"] == 6
    assert report["same_prompt_tokens_match"] is True
    assert report["prompt_tokens_from_llamacpp"] == {
        "prompt": "hello",
        "n_tokens": 3,
        "token_ids": [1, 2, 3],
    }
    assert report["expected_token"] == {
        "token_id": 4,
        "token_text": " expected",
        "llamacpp_logit": 7.0,
        "llamacpp_rank": 2,
        "host_logit": 9.0,
        "host_rank": 1,
    }
    assert report["generated_token"] == {
        "token_id": 2,
        "token_text": "generated",
        "llamacpp_logit": 8.0,
        "llamacpp_rank": 1,
        "host_logit": None,
        "host_rank": None,
    }
    assert report["llamacpp_top_tokens"] == [
        {"rank": 1, "token_id": 2, "logit": 8.0},
        {"rank": 2, "token_id": 4, "logit": 7.0},
        {"rank": 3, "token_id": 3, "logit": 2.5},
    ]
    assert report["comparison"] == {
        "expected_token_id": 4,
        "generated_token_id": 2,
        "llamacpp_top1_token_id": 2,
        "llamacpp_expected_outranks_generated": False,
        "llamacpp_expected_is_top1": False,
        "llamacpp_generated_is_top1": True,
        "host_expected_is_top1": True,
    }
    assert report["raw_outputs_removed_after_parse"] is True
    assert not (tmp_path / "raw" / "llamacpp-Step-3.7-flash-Q3_K_L-00001-of-00003.bin").exists()


def test_stepfun_llamacpp_logits_probe_blocks_when_special_token_flag_is_missing(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(
        fake_debug,
        logits=[0.0, 1.0, 2.0, 3.0, 4.0],
        token_ids=[1, 2, 3],
        help_text="--save-logits --logits-output-dir",
    )
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        execute=True,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["missing_evidence"] == [
        "llama_debug_same_prompt_special_token_support_present"
    ]
    assert report["llama_debug_same_prompt_capability"][
        "same_prompt_special_token_flag_present"
    ] is False
    assert report["blocked_reason"] == (
        "built llama-debug exposes --save-logits but does not accept a special-token "
        "parsing flag, so it cannot reproduce the retained StepFun prompt token IDs"
    )
    assert not (tmp_path / "raw").exists()


def test_stepfun_llamacpp_logits_probe_detects_prompt_token_mismatch(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(fake_debug, logits=[0.0, 1.0, 2.0, 3.0, 4.0], token_ids=[1, 9, 3])
    model.write_text("fake")

    report = build_llamacpp_logits_probe(
        prompt_artifact=prompt,
        llama_debug=fake_debug,
        model=model,
        raw_output_dir=tmp_path / "raw",
        generated_token_id=2,
        execute=True,
    )

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["same_prompt_tokens_match"] is False
    assert report["blocked_reason"] == (
        "llama-debug tokenization did not match retained host prompt input IDs"
    )


def test_stepfun_llamacpp_logits_probe_cli_writes_report(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    output = tmp_path / "probe.json"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(fake_debug, logits=[0.0, 1.0, 2.0, 3.0, 4.0], token_ids=[1, 2, 3])
    model.write_text("fake")

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--llama-debug",
            str(fake_debug),
            "--model",
            str(model),
            "--raw-output-dir",
            str(tmp_path / "raw"),
            "--generated-token-id",
            "2",
            "--execute",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "captured"
    assert payload["same_prompt_tokens_match"] is True


def test_stepfun_llamacpp_logits_probe_cli_compact_modes(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    fake_debug = tmp_path / "llama-debug"
    model = tmp_path / "model.gguf"
    output = tmp_path / "compact.json"
    _write_prompt_artifact(prompt)
    _write_fake_llama_debug(fake_debug, logits=[0.0, 1.0, 8.0, 3.0, 7.0], token_ids=[1, 2, 3])
    model.write_text("fake")
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--llama-debug",
        str(fake_debug),
        "--model",
        str(model),
        "--raw-output-dir",
        str(tmp_path / "raw"),
        "--generated-token-id",
        "2",
        "--execute",
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "captured"
    assert main([*base_args, "--same-prompt-tokens-match-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--expected-outranks-generated-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
