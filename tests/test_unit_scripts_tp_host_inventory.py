"""CPU-only tests for scripts/tp_host_inventory.py parsing and summarization.

No HIP/ROCm is touched: the module-level helpers are pure functions over sysfs
text and command output.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_host_inventory.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp_host_inventory_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_parse_device_list_ordered_and_unique(mod) -> None:
    assert mod.parse_device_list("0,1") == [0, 1]
    assert mod.parse_device_list(" 1 , 0 ") == [1, 0]
    with pytest.raises(ValueError):
        mod.parse_device_list("0,0")
    with pytest.raises(ValueError):
        mod.parse_device_list("0,,1")
    with pytest.raises(ValueError):
        mod.parse_device_list("gpu0")
    with pytest.raises(ValueError):
        mod.parse_device_list("")


def test_pcie_link_parsing(mod) -> None:
    assert mod.parse_pcie_link_speed("16.0 GT/s PCIe") == 16.0
    assert mod.parse_pcie_link_speed("8 GT/s PCIe") == 8.0
    assert mod.parse_pcie_link_speed("n/a") is None
    assert mod.parse_pcie_link_width("16") == 16
    assert mod.parse_pcie_link_width("x8") == 8
    assert mod.parse_pcie_link_width("") is None


def test_pcie_link_snapshot_flags_downgrade(mod, tmp_path: pathlib.Path) -> None:
    (tmp_path / "current_link_speed").write_text("8.0 GT/s PCIe\n", encoding="utf-8")
    (tmp_path / "max_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    (tmp_path / "current_link_width").write_text("16\n", encoding="utf-8")
    (tmp_path / "max_link_width").write_text("16\n", encoding="utf-8")
    snapshot = mod.pcie_link_snapshot(tmp_path)
    assert snapshot["current_speed_gts"] == 8.0
    assert snapshot["max_speed_gts"] == 16.0
    assert snapshot["downgraded"] is True


def test_pcie_link_snapshot_full_rate_is_not_downgraded(mod, tmp_path: pathlib.Path) -> None:
    (tmp_path / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    (tmp_path / "max_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    (tmp_path / "current_link_width").write_text("16\n", encoding="utf-8")
    (tmp_path / "max_link_width").write_text("16\n", encoding="utf-8")
    assert mod.pcie_link_snapshot(tmp_path)["downgraded"] is False


def test_parse_lspci_tree_records_depth_and_slots(mod) -> None:
    text = """-+-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
           +-01.1-[01]----00.0  MAXIO Technology NVMe SSD Controller
           +-03.1-[0b-0d]----00.0-[0c-0d]----00.0-[0d]--+-00.0  AMD/ATI Navi 31 [Radeon Pro W7900]
"""
    entries = mod.parse_lspci_tree(text)
    assert entries[0]["bridge"] is None  # root complex line has no upstream bridge
    assert entries[0]["device"] == "00.0"
    assert entries[0]["bus_ranges"] == []
    nvme = entries[1]
    assert nvme["bridge"] == "01.1"
    assert nvme["bus_ranges"] == ["01"]
    assert nvme["device"] == "00.0"
    gpu = entries[2]
    assert gpu["bridge"] == "03.1"
    assert gpu["bus_ranges"] == ["0b-0d", "0c-0d", "0d"]
    assert gpu["device"] == "00.0"
    assert gpu["depth"] >= 2


def test_parse_acs_entries_extracts_only_requested_slots(mod) -> None:
    text = """\
03:00.0 VGA compatible controller: AMD/ATI Navi 31
        Capabilities: [4c0] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl+ DirectTrans+
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
c3:00.0 VGA compatible controller: AMD/ATI Navi 31
        Capabilities: [4c0] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl+ DirectTrans+
"""
    entries = mod.parse_acs_entries(text, ["03:00.0"])
    assert set(entries) == {"03:00.0"}
    block = entries["03:00.0"]
    assert block["present"] is True
    assert len(block["raw"]) == 3
    # ACSCap advertises the bits; ACSCtl is what is actually in force, and the
    # test slot disables every one of them.
    assert block["capability_lines"] == ["SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl+ DirectTrans+"]
    assert block["control_lines"] == ["SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-"]
    assert block["enabled_bits"] == []
    assert block["redirect_bits_enabled"] == []
    assert block["blocks_peer_dma"] is False


def test_parse_acs_entries_flags_bridge_redirection(mod) -> None:
    """A bridge with ACS redirection enabled blocks P2P DMA below it."""

    text = """\
0c:00.0 PCI bridge: Advanced Micro Devices, Inc. [AMD/ATI] Device 1478
        Capabilities: [2a0 v1] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans+
                ACSCtl: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-
0d:00.0 VGA compatible controller: AMD/ATI Navi 31
        Capabilities: [2a0 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
"""
    entries = mod.parse_acs_entries(text, ["0c:00.0", "0d:00.0"])
    assert entries["0c:00.0"]["blocks_peer_dma"] is True
    # Only the three bits that redirect peer TLPs upstream decide the verdict.
    assert entries["0c:00.0"]["redirect_bits_enabled"] == [
        "CmpltRedir",
        "ReqRedir",
        "UpstreamFwd",
    ]
    # Source validation and translation blocking are reported, not counted:
    # they affect peer routing in specific cases rather than redirecting TLPs.
    assert entries["0c:00.0"]["other_bits_enabled"] == ["SrcValid", "TransBlk"]
    # DirectTrans- in ACSCtl is a disabled capability, never a blocker.
    assert entries["0c:00.0"]["direct_translated_p2p_enabled"] is False
    assert entries["0d:00.0"]["blocks_peer_dma"] is False


def test_collect_acs_state_records_denied_reads(tmp_path: pathlib.Path, mod, monkeypatch) -> None:
    """Unreadable extended capability space must not look like absent ACS."""

    def fake_run(command, timeout: float = 10.0):
        if command[0] == "sudo":
            return {"available": True, "returncode": 1, "stdout": "", "stderr": "password required"}
        return {
            "available": True,
            "returncode": 0,
            "stdout": "0d:00.0 VGA compatible controller: AMD/ATI Navi 31\n\tCapabilities: <access denied>\n",
            "stderr": "",
        }

    monkeypatch.setattr(mod, "_run", fake_run)
    state = mod.collect_acs_state(["0d:00.0"])
    assert state["access_denied"] is True
    assert state["entries"] == {}
    assert "needs root" in state["note"]


def test_parse_iommu_state(tmp_path: pathlib.Path, mod) -> None:
    root = tmp_path / "iommu_groups"
    (root / "0").mkdir(parents=True)
    (root / "12").mkdir(parents=True)
    state = mod.parse_iommu_state(iommu_groups_root=root, cmdline="quiet amd_iommu=on iommu=pt")
    assert state["enabled"] is True
    assert state["group_count"] == 2
    assert state["groups"] == ["0", "12"]
    assert "amd_iommu=on" in state["kernel_cmdline_iommu"]
    assert mod.parse_iommu_state(iommu_groups_root=tmp_path / "missing", cmdline="quiet")["enabled"] is False


def test_parse_rocm_smi_json_normalizes_fields(mod) -> None:
    payload = {
        "card0": {
            "Temperature (Sensor edge) (C)": "43.0",
            "Average Power (W)": "28.123",
            "Max Power (W)": "295.0",
            "sclk clock speed:": "(2100Mhz)",
            "sclk clock level:": "1",
            "mclk clock speed:": "(96Mhz)",
            "Card Series": "Radeon Pro W7900",
            "PCI Bus": "0000:C3:00.0",
            "pcie clock level": "2 (16.0GT/s x8)",
        }
    }
    cards = mod.parse_rocm_smi_json(payload)
    card = cards["card0"]
    assert card["temperature_edge_c"] == 43.0
    assert card["power_average_w"] == 28.123
    assert card["power_cap_w"] == 295.0
    assert card["sclk_mhz"] == 2100.0
    assert card["sclk_raw"] == "(2100Mhz)"
    assert card["pcie_speed_gts"] == 16.0
    assert card["pcie_width_lanes"] == 8
    assert card["mclk_mhz"] == 96.0
    assert card["model"] == "Radeon Pro W7900"
    assert card["pci_bus"] == "0000:C3:00.0"
    assert mod.parse_rocm_smi_json(None) == {}
    assert mod.parse_rocm_smi_json({"card0": "not-a-dict"}) == {}


def test_summarize_host_and_device_shape(mod) -> None:
    host = mod.summarize_host(
        hostname="w7900",
        uname={"system": "Linux", "release": "7.1.3", "machine": "x86_64", "node": "w7900"},
        cpu_model="AMD Ryzen 9 5950X",
        cpu_count=32,
        affinity=[0, 1, 2],
        mem_total_bytes=64 << 30,
        numa_nodes=[0],
    )
    assert host["process_cpu_affinity"] == [0, 1, 2]
    assert host["mem_total_bytes"] == 64 << 30

    device = mod.summarize_device(
        rank=1,
        index=1,
        name="AMD Radeon RX 7900 XTX",
        arch="gfx1100",
        uuid="0000-1111",
        uuid_hex="00" * 16,
        pci_bus_id="0000:03:00.0",
        vram_total_bytes=24 << 30,
        vram_free_bytes=23 << 30,
        numa_node=0,
        pcie={"current_speed_gts": 16.0},
        smi={"sclk_mhz": 2100.0},
    )
    assert device["rank"] == 1
    assert device["hip_index"] == 1
    assert device["pcie"]["current_speed_gts"] == 16.0
    assert device["rocm_smi"]["sclk_mhz"] == 2100.0


def test_inventory_is_json_serializable(mod, tmp_path: pathlib.Path) -> None:
    (tmp_path / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
    inventory = {
        "kind": "tp_host_inventory",
        "devices": [mod.summarize_device(
            rank=0,
            index=0,
            name="W7900",
            arch="gfx1100",
            uuid=None,
            uuid_hex=None,
            pci_bus_id="0000:c3:00.0",
            vram_total_bytes=1,
            vram_free_bytes=1,
            numa_node=0,
            pcie=mod.pcie_link_snapshot(tmp_path),
            smi=None,
        )],
    }
    assert json.loads(json.dumps(inventory))["devices"][0]["rank"] == 0


def test_collect_display_load_reports_openers(tmp_path: pathlib.Path, mod) -> None:
    """A process holding a DRM node must be visible; an idle GPU must read empty."""

    drm = tmp_path / "dri"
    drm.mkdir()
    for name in ("card0", "renderD128"):
        (drm / name).write_text("", encoding="utf-8")
    (drm / "by-path").mkdir()

    proc = tmp_path / "proc"
    busy_pid = proc / "4242"
    (busy_pid / "fd").mkdir(parents=True)
    (busy_pid / "comm").write_text("compositor\n", encoding="utf-8")
    (busy_pid / "fd" / "7").symlink_to(drm / "card0")

    idle_pid = proc / "4343"
    (idle_pid / "fd").mkdir(parents=True)
    (idle_pid / "fd" / "3").symlink_to("/dev/null")

    # ``is_char_device`` is only true for real device nodes, so exercise the
    # opener scan with a monkeypatched predicate rather than a fake node.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod.Path, "is_char_device", lambda self: True, raising=False)
    try:
        state = mod.collect_display_load(proc_root=proc, drm_root=drm)
    finally:
        monkeypatch.undo()

    assert state["devices"] == ["card0", "renderD128"]
    assert state["busy"] is True
    assert [entry["comm"] for entry in state["openers"]["card0"]] == ["compositor"]
    assert state["openers"]["renderD128"] == []
    assert all(entry["pid"] != os.getpid() for entry in state["openers"]["card0"])


def test_collect_display_load_handles_missing_drm(tmp_path: pathlib.Path, mod) -> None:
    state = mod.collect_display_load(proc_root=tmp_path / "proc", drm_root=tmp_path / "dri")
    assert state["devices"] == []
    assert state["busy"] is False


def test_acs_direct_translated_p2p_is_not_a_blocker(mod) -> None:
    """DirectTrans+ enables translated peer requests; it is not a redirect.

    Treating it as a blanket blocker would report a P2P-capable bridge as
    blocking on the strength of a capability bit.
    """

    text = """\
0c:00.0 PCI bridge: Advanced Micro Devices, Inc. [AMD/ATI] Device 1478
        Capabilities: [2a0 v1] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans+
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans+
"""
    block = mod.parse_acs_entries(text, ["0c:00.0"])["0c:00.0"]
    assert block["direct_translated_p2p_enabled"] is True
    assert block["blocks_peer_dma"] is False
    assert block["redirect_bits_enabled"] == []


def test_display_load_records_every_drm_node_a_process_holds(tmp_path: pathlib.Path, mod) -> None:
    """A compositor holding both cards must not hide one of them.

    Stopping at the first DRM file descriptor of each process reported the
    second card as idle.
    """

    proc = tmp_path / "proc"
    drm = tmp_path / "dri"
    drm.mkdir()
    for name in ("card0", "card1"):
        (drm / name).write_text("", encoding="utf-8")
    pid_dir = proc / "4242"
    (pid_dir / "fd").mkdir(parents=True)
    (pid_dir / "comm").write_text("compositor\n", encoding="utf-8")
    (pid_dir / "fd" / "3").symlink_to(str(drm / "card0"))
    (pid_dir / "fd" / "4").symlink_to(str(drm / "card1"))

    # ``is_char_device`` is only true for real device nodes, so exercise the
    # opener scan with a monkeypatched predicate rather than a fake node.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod.Path, "is_char_device", lambda self: True, raising=False)
    try:
        record = mod.collect_display_load(proc_root=proc, drm_root=drm)
    finally:
        monkeypatch.undo()

    assert record["openers"]["card0"] == [{"pid": 4242, "comm": "compositor"}]
    assert record["openers"]["card1"] == [{"pid": 4242, "comm": "compositor"}]
    assert record["busy"] is True
    assert record["visibility_complete"] is True
    assert record["unreadable_fd_tables"] == []


def test_display_load_records_incomplete_visibility(tmp_path: pathlib.Path, mod) -> None:
    """An unreadable fd table means no opener observed, not no opener.

    The screen must not present an empty result as proof that no compositor
    shares either card.
    """

    proc = tmp_path / "proc"
    drm = tmp_path / "dri"
    drm.mkdir()
    (drm / "card0").write_text("", encoding="utf-8")
    (proc / "9999").mkdir(parents=True)  # no fd directory at all

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod.Path, "is_char_device", lambda self: True, raising=False)
    try:
        record = mod.collect_display_load(proc_root=proc, drm_root=drm)
    finally:
        monkeypatch.undo()

    assert record["openers"]["card0"] == []
    assert record["busy"] is False
    assert record["visibility_complete"] is False
    assert record["unreadable_fd_tables"] == [9999]


# -- per-device architecture attribution --------------------------------------


def test_per_device_arch_probe_keeps_one_entry_per_gpu() -> None:
    """Deduplicating the probe output destroys the per-device information.

    ``amdgpu-arch`` prints one architecture per visible GPU in HIP device order.
    A deduplicated list indexed by rank position mislabels every device on a
    mixed-architecture host or a reordered device selection.
    """

    from hipengine.kernels.backends import _parse_arches, _parse_arches_per_device

    text = "gfx1100\ngfx1030\ngfx1100\n"
    assert _parse_arches(text) == ("gfx1100", "gfx1030")
    assert _parse_arches_per_device(text) == ("gfx1100", "gfx1030", "gfx1100")


def test_device_arch_is_selected_by_hip_index_not_rank(mod) -> None:
    """The index is the HIP device index, not a position in a rank list."""

    per_device = ("gfx1030", "gfx1100")
    assert mod.arch_for_device(per_device, 0) == "gfx1030"
    assert mod.arch_for_device(per_device, 1) == "gfx1100"
    # A plan may order devices [1, 0]: rank 0 is then HIP device 1.
    plan_order = [1, 0]
    assert [mod.arch_for_device(per_device, index) for index in plan_order] == [
        "gfx1100",
        "gfx1030",
    ]


def test_device_arch_is_unknown_when_the_probe_did_not_cover_it(mod) -> None:
    """A device the probe missed reports nothing instead of another device's arch."""

    assert mod.arch_for_device(("gfx1100",), 3) is None
    assert mod.arch_for_device((), 0) is None
    assert mod.arch_for_device(("gfx1100",), -1) is None
