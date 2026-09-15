"""VibeVoice-TTS checkpoint weight inventory.

Milestone 1 requires that every checkpoint weight is inventoried and accounted
for. This script reads the safetensors headers (shapes and dtypes only -- it does
not materialise 5.4 GB of weights) and reports, per model component, the tensor
count, parameter count and byte count, plus a total.

With ``--verify-model`` it also loads the checkpoint through the fork and checks
the two ways a weight can go unaccounted for:

- **orphans**: a key present in the checkpoint that the model never consumes;
- **missing**: a model parameter absent from the checkpoint.

``missing`` is expected to contain exactly ``lm_head.weight``, which the model
ties to ``model.language_model.embed_tokens.weight`` when
``tie_word_embeddings`` is set. The script asserts that tie is real storage
sharing rather than a same-shaped copy, because a silent untie would change the
arithmetic while leaving the key set unchanged.

Usage:
    python3 scripts/vibevoice_tts_weight_inventory.py \\
        [--out tests/fixtures/vibevoice_tts/weight_inventory.json]

    # with the orphan/missing/tie cross-check (needs the oracle venv)
    PYTHONPATH=/home/lhl/VibeVoice-community \\
        /home/lhl/venvs/vibevoice-tts-oracle/bin/python \\
        scripts/vibevoice_tts_weight_inventory.py --verify-model
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_MODEL = "microsoft/VibeVoice-1.5B"
DEFAULT_OUT = Path("tests/fixtures/vibevoice_tts/weight_inventory.json")

# Component order is fixed so the artifact is stable across runs.
COMPONENT_ORDER = [
    "model.language_model",
    "model.acoustic_tokenizer",
    "model.semantic_tokenizer",
    "model.prediction_head",
    "model.semantic_connector",
    "model.acoustic_connector",
    "model.speech_scaling_factor",
    "model.speech_bias_factor",
]

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def _snapshot_dir(model_id: str) -> Path:
    pattern = Path.home() / f".cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*"
    snapshots = sorted(glob.glob(str(pattern)))
    if not snapshots:
        raise SystemExit(f"no local snapshot found for {model_id}; snapshot_download it first")
    return Path(snapshots[-1])


def _component_of(key: str) -> str:
    """Map a tensor key to its owning component (``model.<component>``)."""
    parts = key.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else key


def build_inventory(snap: Path) -> dict:
    from safetensors import safe_open

    index_path = snap / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"no safetensors index at {index_path}")

    weight_map = json.loads(index_path.read_text())["weight_map"]

    comps: dict[str, dict] = defaultdict(
        lambda: {"tensors": 0, "params": 0, "bytes": 0, "dtypes": set()}
    )
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(snap / shard), framework="pt") as handle:
            for key in handle.keys():
                sl = handle.get_slice(key)
                shape = tuple(sl.get_shape())
                dtype = sl.get_dtype()
                n = 1
                for d in shape:
                    n *= d
                c = comps[_component_of(key)]
                c["tensors"] += 1
                c["params"] += n
                c["bytes"] += n * DTYPE_BYTES.get(dtype, 2)
                c["dtypes"].add(dtype)

    components = []
    for name in COMPONENT_ORDER:
        if name not in comps:
            continue
        c = comps[name]
        components.append({
            "component": name,
            "tensors": c["tensors"],
            "params": c["params"],
            "bytes": c["bytes"],
            "gib": round(c["bytes"] / (1024 ** 3), 4),
            "dtypes": sorted(c["dtypes"]),
        })

    unexpected = sorted(set(comps) - set(COMPONENT_ORDER))
    if unexpected:
        raise SystemExit(f"unexpected components not in COMPONENT_ORDER: {unexpected}")

    total_tensors = sum(c["tensors"] for c in components)
    total_params = sum(c["params"] for c in components)
    total_bytes = sum(c["bytes"] for c in components)

    return {
        "model_id": DEFAULT_MODEL,
        "model_revision": snap.name,
        "total_tensors": total_tensors,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 4),
        "components": components,
    }


def verify_model(snap: Path, inventory: dict) -> dict:
    """Cross-check the header inventory against the model the fork actually builds."""
    import torch
    from safetensors import safe_open

    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    weight_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    ckpt_keys = set(weight_map)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        DEFAULT_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    state_keys = set(model.state_dict().keys())

    orphan = sorted(ckpt_keys - state_keys)
    missing = sorted(state_keys - ckpt_keys)

    tied_ok = None
    if "lm_head.weight" in missing:
        tied_ok = (
            model.lm_head.weight.data_ptr()
            == model.model.language_model.embed_tokens.weight.data_ptr()
        )
        if not tied_ok:
            raise SystemExit(
                "lm_head.weight is absent from the checkpoint but is NOT tied to "
                "embed_tokens.weight; the weight would be silently random"
            )

    live_params = sum(p.numel() for p in model.parameters())
    # The two scaling factors are registered with ``register_buffer``, so they are
    # in the state dict and on disk but absent from ``.parameters()``. Comparing
    # against ``.parameters()`` therefore misses them by exactly 2. Count the
    # state dict instead, excluding the tied keys: those are the entries the
    # checkpoint does not store separately.
    state = model.state_dict()
    stored_params = sum(v.numel() for k, v in state.items() if k not in set(missing))
    header_params = inventory["total_params"]

    result = {
        "checkpoint_keys": len(ckpt_keys),
        "model_state_keys": len(state_keys),
        "orphan_keys": orphan,
        "missing_keys": missing,
        "lm_head_tied_to_embed_tokens": tied_ok,
        "parameter_count": live_params,
        "buffer_parameter_delta": stored_params - live_params,
        "stored_parameter_count": stored_params,
        "header_parameter_count": header_params,
        "parameter_counts_agree": stored_params == header_params,
    }
    if orphan:
        raise SystemExit(f"checkpoint keys the model never consumes: {orphan}")
    if not result["parameter_counts_agree"]:
        raise SystemExit(
            f"parameter mismatch: state dict reports {stored_params} stored, "
            f"headers report {header_params}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-model", action="store_true",
                        help="load the fork's model and check orphans/missing/ties")
    args = parser.parse_args()

    snap = _snapshot_dir(args.model)
    inv = build_inventory(snap)
    inv["model_id"] = args.model

    print(f"{'component':32s} {'tensors':>8s} {'params':>16s} {'GiB':>8s}")
    for c in inv["components"]:
        print(f"{c['component']:32s} {c['tensors']:8d} {c['params']:16,d} {c['gib']:8.3f}")
    print(f"{'TOTAL':32s} {inv['total_tensors']:8d} {inv['total_params']:16,d} {inv['total_gib']:8.3f}")

    if args.verify_model:
        print("\nverifying against the fork's model ...")
        inv["model_cross_check"] = verify_model(snap, inv)
        cc = inv["model_cross_check"]
        print(f"  checkpoint keys           {cc['checkpoint_keys']}")
        print(f"  model state keys          {cc['model_state_keys']}")
        print(f"  orphan keys               {len(cc['orphan_keys'])}")
        print(f"  missing keys              {cc['missing_keys']}")
        print(f"  lm_head tied to embed     {cc['lm_head_tied_to_embed_tokens']}")
        print(f"  stored parameters         {cc['stored_parameter_count']:,}")
        print(f"  parameters() only         {cc['parameter_count']:,} (buffers excluded)")
        print(f"  parameter counts agree    {cc['parameter_counts_agree']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inv, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
