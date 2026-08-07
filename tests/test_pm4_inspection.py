from __future__ import annotations

import ctypes
import hashlib
import struct
from pathlib import Path

import pytest

import hipengine.core.pm4.graph as pm4_graph
from hipengine.core.pm4 import (
    HipDim3,
    HipKernelNodeParams,
    LaunchContext,
    Pm4InspectionError,
    extract_elf_section,
    inspect_hip_graph,
    pack_kernargs,
    parse_amdgpu_kernels,
    select_amdgpu_code_object,
    topological_order,
)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _pack_msgpack(value: object) -> bytes:
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int):
        if 0 <= value <= 0x7F:
            return bytes((value,))
        if -32 <= value < 0:
            return bytes((value & 0xFF,))
        if 0 <= value <= 0xFF:
            return b"\xcc" + struct.pack(">B", value)
        if 0 <= value <= 0xFFFF:
            return b"\xcd" + struct.pack(">H", value)
        if 0 <= value <= 0xFFFFFFFF:
            return b"\xce" + struct.pack(">I", value)
        if value >= 0:
            return b"\xcf" + struct.pack(">Q", value)
        return b"\xd3" + struct.pack(">q", value)
    if isinstance(value, str):
        encoded = value.encode()
        if len(encoded) < 32:
            return bytes((0xA0 | len(encoded),)) + encoded
        return b"\xd9" + struct.pack(">B", len(encoded)) + encoded
    if isinstance(value, list):
        assert len(value) < 16
        return bytes((0x90 | len(value),)) + b"".join(_pack_msgpack(item) for item in value)
    if isinstance(value, dict):
        assert len(value) < 16
        return bytes((0x80 | len(value),)) + b"".join(
            _pack_msgpack(key) + _pack_msgpack(item) for key, item in value.items()
        )
    raise TypeError(type(value))


def _metadata_blob(*, unknown_hidden: bool = False) -> bytes:
    args = [
        {".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
        {".offset": 8, ".size": 4, ".value_kind": "by_value"},
        {".offset": 16, ".size": 4, ".value_kind": "hidden_block_count_x"},
        {".offset": 20, ".size": 2, ".value_kind": "hidden_group_size_x"},
        {
            ".offset": 22,
            ".size": 2,
            ".value_kind": "hidden_unrecognized" if unknown_hidden else "hidden_remainder_x",
        },
        {".offset": 24, ".size": 8, ".value_kind": "hidden_global_offset_x"},
        {".offset": 32, ".size": 2, ".value_kind": "hidden_grid_dims"},
    ]
    return _pack_msgpack(
        {
            "amdhsa.version": [1, 2],
            "amdhsa.target": "amdgcn-amd-amdhsa--gfx1100",
            "amdhsa.kernels": [
                {
                    ".name": "test_kernel",
                    ".symbol": "test_kernel.kd",
                    ".kernarg_segment_size": 40,
                    ".kernarg_segment_align": 8,
                    ".group_segment_fixed_size": 0,
                    ".private_segment_fixed_size": 0,
                    ".uses_dynamic_stack": False,
                    ".wavefront_size": 32,
                    ".args": args,
                }
            ],
        }
    )


def _note(metadata: bytes, *, owner: bytes = b"AMDGPU\0", note_type: int = 32) -> bytes:
    data = bytearray(struct.pack("<III", len(owner), len(metadata), note_type))
    data.extend(owner)
    data.extend(b"\0" * (_align(len(data), 4) - len(data)))
    data.extend(metadata)
    data.extend(b"\0" * (_align(len(data), 4) - len(data)))
    return bytes(data)


def _elf_with_sections(sections: list[tuple[str, int, bytes]]) -> bytes:
    """Build a bounded little-endian ELF64 fixture with named sections."""

    names = bytearray(b"\0")
    name_offsets: dict[str, int] = {}
    for name, _, _ in [*sections, (".shstrtab", 3, b"")]:
        name_offsets[name] = len(names)
        names.extend(name.encode() + b"\0")

    payloads = [*sections, (".shstrtab", 3, bytes(names))]
    image = bytearray(64)
    entries: list[tuple[int, int, int, int]] = []
    for name, section_type, payload in payloads:
        offset = _align(len(image), 8)
        image.extend(b"\0" * (offset - len(image)))
        image.extend(payload)
        entries.append((name_offsets[name], section_type, offset, len(payload)))

    section_offset = _align(len(image), 8)
    image.extend(b"\0" * (section_offset - len(image)))
    image.extend(bytes(64))  # null section
    for name_offset, section_type, offset, size in entries:
        header = bytearray(64)
        struct.pack_into("<II", header, 0, name_offset, section_type)
        struct.pack_into("<QQ", header, 24, offset, size)
        struct.pack_into("<Q", header, 48, 1)
        image.extend(header)

    image[0:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHI", image, 16, 3, 224, 1)
    struct.pack_into("<Q", image, 40, section_offset)
    struct.pack_into("<HHHHHH", image, 52, 64, 0, 0, 64, len(entries) + 1, len(entries))
    return bytes(image)


def _hsaco(metadata: bytes | None = None) -> bytes:
    metadata = _metadata_blob() if metadata is None else metadata
    return _elf_with_sections([(".note", 7, _note(metadata))])


def _bundle(entries: list[tuple[str, bytes]]) -> bytes:
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    toc_size = len(magic) + 8 + sum(24 + len(name.encode()) for name, _ in entries)
    cursor = toc_size
    descriptors: list[bytes] = []
    payloads: list[bytes] = []
    for name, payload in entries:
        encoded = name.encode()
        descriptors.append(struct.pack("<QQQ", cursor, len(payload), len(encoded)) + encoded)
        payloads.append(payload)
        cursor += len(payload)
    return magic + struct.pack("<Q", len(entries)) + b"".join(descriptors) + b"".join(payloads)


def _dso(fatbin: bytes) -> bytes:
    return _elf_with_sections([(".hip_fatbin", 1, fatbin)])


def test_dso_cache_reuses_only_an_unchanged_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cached.so"
    data = _dso(_bundle([("hipv4-amdgcn-amd-amdhsa--gfx1100", _hsaco())]))
    path.write_bytes(data)
    pm4_graph.clear_dso_cache_for_tests()
    reads = 0
    real_read_stable = pm4_graph._read_stable

    def counted_read_stable(candidate: Path) -> bytes:
        nonlocal reads
        reads += 1
        return real_read_stable(candidate)

    monkeypatch.setattr(pm4_graph, "_read_stable", counted_read_stable)
    first = pm4_graph._load_dso(path, "gfx1100")
    second = pm4_graph._load_dso(path, "gfx1100")
    assert first is second
    assert reads == 1

    path.write_bytes(data)
    third = pm4_graph._load_dso(path, "gfx1100")
    assert third is not first
    assert reads == 2
    pm4_graph.clear_dso_cache_for_tests()


def test_extracts_exact_fatbin_and_selects_exact_gfx_target() -> None:
    wanted = _hsaco()
    other = _hsaco(_pack_msgpack({"amdhsa.kernels": []}))
    fatbin = _bundle(
        [
            ("host-x86_64-unknown-linux-gnu-", b""),
            ("hipv4-amdgcn-amd-amdhsa--gfx1151", other),
            ("hipv4-amdgcn-amd-amdhsa--gfx1100:sramecc-", wanted),
        ]
    )

    extracted = extract_elf_section(_dso(fatbin), ".hip_fatbin")
    selected = select_amdgpu_code_object(extracted, "gfx1100")

    assert selected.target_id.endswith("gfx1100:sramecc-")
    assert selected.image == wanted
    assert selected.sha256 == hashlib.sha256(wanted).hexdigest()


def test_bundle_target_selection_rejects_missing_ambiguous_and_malformed() -> None:
    hsaco = _hsaco()
    with pytest.raises(Pm4InspectionError, match="no AMDGPU code object"):
        select_amdgpu_code_object(_bundle([("host-x86_64", b"")]), "gfx1100")
    with pytest.raises(Pm4InspectionError, match="multiple AMDGPU code objects"):
        select_amdgpu_code_object(
            _bundle(
                [
                    ("hipv4-amdgcn-amd-amdhsa--gfx1100", hsaco),
                    ("hipv4-amdgcn-amd-amdhsa--gfx1100:xnack-", hsaco),
                ]
            ),
            "gfx1100",
        )
    with pytest.raises(Pm4InspectionError, match="truncated|range"):
        select_amdgpu_code_object(b"__CLANG_OFFLOAD_BUNDLE__\x01", "gfx1100")


def test_parses_bounded_amdgpu_metadata_and_rejects_bad_layout() -> None:
    kernels = parse_amdgpu_kernels(_hsaco())
    kernel = kernels["test_kernel.kd"]

    assert kernel.name == "test_kernel"
    assert kernel.kernarg_size == 40
    assert kernel.kernarg_align == 8
    assert kernel.private_segment_size == 0
    assert kernel.wavefront_size == 32
    assert [field.value_kind for field in kernel.args[:2]] == ["global_buffer", "by_value"]

    overlapping = _pack_msgpack(
        {
            "amdhsa.kernels": [
                {
                    ".name": "bad",
                    ".symbol": "bad.kd",
                    ".kernarg_segment_size": 8,
                    ".args": [
                        {".offset": 0, ".size": 8, ".value_kind": "by_value"},
                        {".offset": 4, ".size": 4, ".value_kind": "by_value"},
                    ],
                }
            ]
        }
    )
    with pytest.raises(Pm4InspectionError, match="overlap"):
        parse_amdgpu_kernels(_hsaco(overlapping))

    deeply_nested = bytes((0x91,)) * 40 + b"\x00"
    with pytest.raises(Pm4InspectionError, match="MessagePack"):
        parse_amdgpu_kernels(_hsaco(deeply_nested))


def test_packs_explicit_and_allowlisted_hidden_kernargs_exactly() -> None:
    kernel = parse_amdgpu_kernels(_hsaco())["test_kernel.kd"]
    pointer = 0x123456789ABCDEF0
    scalar = 37
    context = LaunchContext(grid_blocks=(7, 2, 1), block=(256, 1, 1), dynamic_shared_bytes=0)

    packed = pack_kernargs(
        kernel,
        [pointer.to_bytes(8, "little"), scalar.to_bytes(4, "little")],
        context,
    )

    assert len(packed) == 40
    assert int.from_bytes(packed[0:8], "little") == pointer
    assert int.from_bytes(packed[8:12], "little") == scalar
    assert packed[12:16] == bytes(4)
    assert int.from_bytes(packed[16:20], "little") == 7
    assert int.from_bytes(packed[20:22], "little") == 256
    assert int.from_bytes(packed[22:24], "little") == 0
    assert int.from_bytes(packed[24:32], "little") == 0
    assert int.from_bytes(packed[32:34], "little") == 2
    assert packed[34:40] == bytes(6)

    with pytest.raises(Pm4InspectionError, match="explicit argument count"):
        pack_kernargs(kernel, [pointer.to_bytes(8, "little")], context)

    unknown = parse_amdgpu_kernels(_hsaco(_metadata_blob(unknown_hidden=True)))["test_kernel.kd"]
    with pytest.raises(Pm4InspectionError, match="unsupported hidden kernarg"):
        pack_kernargs(
            unknown,
            [pointer.to_bytes(8, "little"), scalar.to_bytes(4, "little")],
            context,
        )


def test_topological_order_is_deterministic_and_rejects_invalid_graphs() -> None:
    assert topological_order((30, 10, 20), ((10, 30), (20, 30))) == (10, 20, 30)
    assert topological_order((30, 10, 20), ()) == (10, 20, 30)

    with pytest.raises(Pm4InspectionError, match="cycle"):
        topological_order((1, 2), ((1, 2), (2, 1)))
    with pytest.raises(Pm4InspectionError, match="unknown node"):
        topological_order((1, 2), ((1, 3),))
    with pytest.raises(Pm4InspectionError, match="duplicate node"):
        topological_order((1, 1), ())


def test_inspects_fake_kernel_graph_into_immutable_exact_manifest(tmp_path: Path) -> None:
    hsaco = _hsaco()
    dso_path = tmp_path / "kernel.so"
    dso_bytes = _dso(
        _bundle(
            [
                ("host-x86_64-unknown-linux-gnu-", b""),
                ("hipv4-amdgcn-amd-amdhsa--gfx1100", hsaco),
            ]
        )
    )
    dso_path.write_bytes(dso_bytes)

    arg0 = ctypes.c_uint64(0x123456789ABCDEF0)
    arg1 = ctypes.c_uint32(37)
    params_array = (ctypes.c_void_p * 2)(ctypes.addressof(arg0), ctypes.addressof(arg1))
    params = HipKernelNodeParams(
        HipDim3(256, 1, 1),
        ctypes.POINTER(ctypes.c_void_p)(),
        ctypes.c_void_p(0xABC0),
        HipDim3(7, 2, 1),
        ctypes.cast(params_array, ctypes.POINTER(ctypes.c_void_p)),
        0,
    )

    class FakeRuntime:
        name_calls = 0

        def graph_nodes(self, graph: int) -> tuple[int, ...]:
            assert graph == 0xCAFE
            return (0x11,)

        def graph_edges(self, graph: int) -> tuple[tuple[int, int], ...]:
            assert graph == 0xCAFE
            return ()

        def graph_node_type(self, node: int) -> int:
            assert node == 0x11
            return 0

        def graph_kernel_node_params(self, node: int) -> HipKernelNodeParams:
            assert node == 0x11
            return params

        def kernel_name_ref_by_ptr(self, function: int, stream: int = 0) -> str:
            assert function == 0xABC0
            self.name_calls += 1
            return "test_kernel"

    runtime = FakeRuntime()
    timings: dict[str, int] = {}
    manifest = inspect_hip_graph(
        runtime,
        0xCAFE,
        gfx_arch="gfx1100",
        stream=0,
        dso_resolver=lambda function: dso_path,
        timings=timings,
    )

    assert timings["total_ns"] > 0
    assert timings["dso_load_ns"] > 0
    assert manifest.graph_handle == 0xCAFE
    assert runtime.name_calls == 1
    assert manifest.order == (0x11,)
    assert manifest.edges == ()
    assert len(manifest.nodes) == 1
    node = manifest.nodes[0]
    assert node.name == "test_kernel"
    assert node.loader_symbol == "test_kernel.kd"
    assert node.grid_blocks == (7, 2, 1)
    assert node.grid_workitems == (1792, 2, 1)
    assert node.block == (256, 1, 1)
    assert node.kernarg[0:8] == arg0.value.to_bytes(8, "little")
    assert node.kernarg[8:12] == arg1.value.to_bytes(4, "little")
    assert node.hsaco == hsaco
    assert node.hsaco_sha256 == hashlib.sha256(hsaco).hexdigest()
    assert node.dso_sha256 == hashlib.sha256(dso_bytes).hexdigest()
    assert len(manifest.fingerprint) == 64

    mutated_params = HipKernelNodeParams(
        HipDim3(256, 1, 1),
        ctypes.POINTER(ctypes.c_void_p)(),
        ctypes.c_void_p(0xABC0),
        HipDim3(8, 2, 1),
        ctypes.cast(params_array, ctypes.POINTER(ctypes.c_void_p)),
        0,
    )

    class MutatingRuntime(FakeRuntime):
        calls = 0

        def graph_kernel_node_params(self, node: int) -> HipKernelNodeParams:
            self.calls += 1
            return params if self.calls == 1 else mutated_params

    with pytest.raises(Pm4InspectionError, match="changed during inspection"):
        inspect_hip_graph(
            MutatingRuntime(),
            0xCAFE,
            gfx_arch="gfx1100",
            stream=0,
            dso_resolver=lambda function: dso_path,
        )


def test_inspector_fails_closed_on_non_kernel_node(tmp_path: Path) -> None:
    class FakeRuntime:
        def graph_nodes(self, graph: int) -> tuple[int, ...]:
            return (1,)

        def graph_edges(self, graph: int) -> tuple[tuple[int, int], ...]:
            return ()

        def graph_node_type(self, node: int) -> int:
            return 1  # memcpy

    with pytest.raises(Pm4InspectionError, match="unsupported HIP graph node type"):
        inspect_hip_graph(FakeRuntime(), 1, gfx_arch="gfx1100", dso_resolver=lambda _: tmp_path)
