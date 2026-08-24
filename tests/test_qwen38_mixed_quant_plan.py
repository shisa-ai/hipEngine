from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qwen38_mixed_quant_plan as plan


def _tensor(name: str, qtype: str, shape: tuple[int, ...] = (256,)):
    return SimpleNamespace(name=name, ggml_type_name=qtype, shape=shape)


def _inventories(*, weak_nextn: bool = False):
    names = list(plan.NEXTN_MINIMUMS)
    names.extend(f"blk.{index}.synthetic_{index}.weight" for index in range(866 - len(names)))
    source = {name: _tensor(name, "BF16") for name in names}
    donor = {}
    for name in names:
        qtype = plan.NEXTN_MINIMUMS.get(name, "IQ4_XS")
        if weak_nextn and name == "blk.64.nextn.eh_proj.weight":
            qtype = "Q4_K"
        donor[name] = _tensor(name, qtype)
    return source, donor


def test_native_projection_promotes_only_declared_unsupported_types() -> None:
    assert plan._project_type("IQ4_XS") == "Q4_K"
    assert plan._project_type("IQ4_NL") == "Q4_K"
    assert plan._project_type("Q3_K") == "Q4_K"
    assert plan._project_type("IQ3_S") == "Q4_K"
    assert plan._project_type("Q6_K") == "Q6_K"
    with pytest.raises(ValueError, match="no declared native projection"):
        plan._project_type("IQ2_XS")


def test_tensor_type_file_uses_exact_escaped_regex(tmp_path: Path) -> None:
    output = tmp_path / "types.txt"
    plan.write_tensor_type_file(
        {"output_types": {"blk.64.nextn.eh_proj.weight": "Q6_K"}},
        output,
    )
    assert output.read_text() == r"^blk\.64\.nextn\.eh_proj\.weight$=Q6_K" + "\n"


def test_build_plan_binds_full_inventory_budget_and_nextn_floors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, donor = _inventories()
    template_info = SimpleNamespace(
        tensors=tuple(donor.values()),
        metadata={
            "quantize.imatrix.dataset": "nonbenchmark-calibration",
            "quantize.imatrix.chunks_count": 45,
            "quantize.imatrix.entries_count": 496,
        },
    )
    monkeypatch.setattr(plan, "_inventory", lambda _paths: (source, {"general.base_model.0.repo_url": "repo"}))
    monkeypatch.setattr(plan, "load_gguf_index", lambda _path: template_info)
    monkeypatch.setattr(plan, "nbytes_for_shape", lambda shape, _qtype: 128)
    monkeypatch.setattr(plan, "_sha256", lambda _path: "a" * 64)
    imatrix = tmp_path / "imatrix.gguf"
    imatrix.write_bytes(b"x")

    manifest = plan.build_plan(
        source_paths=(tmp_path / "source.gguf",),
        template_path=tmp_path / "template.gguf",
        imatrix_path=imatrix,
        source_revision="source-rev",
        template_revision="template-rev",
        source_sha256=("a" * 64,),
        template_sha256="a" * 64,
        imatrix_sha256="a" * 64,
        target_bpw=4.0,
        hard_cap_bpw=8.0,
        hash_sources=False,
    )

    assert manifest["source"]["tensor_count"] == 866
    assert manifest["budget"]["projected_type_counts"] == {
        "Q4_K": 857,
        "Q6_K": 7,
        "Q8_0": 2,
    }
    assert manifest["projection"]["nextn_minimums"] == dict(
        sorted(plan.NEXTN_MINIMUMS.items())
    )
    assert len(manifest["output_types"]) == 866
    assert len(manifest["tensor_inventory_sha256"]) == 64
    assert len(manifest["output_type_manifest_sha256"]) == 64
    assert manifest["calibration"]["benchmark_prompt_overlap_allowed"] is False


def test_build_plan_rejects_a_weakened_nextn_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, donor = _inventories(weak_nextn=True)
    monkeypatch.setattr(plan, "_inventory", lambda _paths: (source, {}))
    monkeypatch.setattr(
        plan,
        "load_gguf_index",
        lambda _path: SimpleNamespace(tensors=tuple(donor.values()), metadata={}),
    )
    monkeypatch.setattr(plan, "nbytes_for_shape", lambda shape, _qtype: 128)
    monkeypatch.setattr(plan, "_sha256", lambda _path: "a" * 64)
    imatrix = tmp_path / "imatrix.gguf"
    imatrix.write_bytes(b"x")

    with pytest.raises(ValueError, match="precision floor failed"):
        plan.build_plan(
            source_paths=(tmp_path / "source.gguf",),
            template_path=tmp_path / "template.gguf",
            imatrix_path=imatrix,
            source_revision="source-rev",
            template_revision="template-rev",
            source_sha256=("a" * 64,),
            template_sha256="a" * 64,
            imatrix_sha256="a" * 64,
            target_bpw=4.0,
            hard_cap_bpw=8.0,
            hash_sources=False,
        )
