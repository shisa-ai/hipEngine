from __future__ import annotations

import struct

import pytest

from hipengine.core.pm4 import Pm4InspectionError
from hipengine.core.pm4.packets import (
    ENABLE_SGPR_KERNARG_SEGMENT_PTR,
    ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER,
    ENABLE_WAVEFRONT_SIZE32,
    DispatchGeometry,
    Gfx1100KernelImage,
    acquire_system,
    dependency_global,
    dependency_local_cache,
    encode_gfx1100_graph,
    packet3,
    vendor_pm4_ib_packet,
    wait_compute_idle,
)


def _image(**updates: int | bool) -> Gfx1100KernelImage:
    values: dict[str, int | bool] = {
        "code_entry": 0x10000,
        "compute_pgm_rsrc1": 0x11,
        "compute_pgm_rsrc2": 0x22,
        "compute_pgm_rsrc3": 0x33,
        "group_segment_size": 0,
        "private_segment_size": 0,
        "dynamic_callstack": False,
        "wave32": True,
        "kernel_code_properties": (
            ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
            | ENABLE_SGPR_KERNARG_SEGMENT_PTR
            | ENABLE_WAVEFRONT_SIZE32
        ),
    }
    values.update(updates)
    return Gfx1100KernelImage(**values)  # type: ignore[arg-type]


def test_gfx1100_single_dispatch_words_are_exact() -> None:
    words = encode_gfx1100_graph(
        [
            (
                _image(),
                DispatchGeometry(grid_workitems=(256, 1, 1), block=(256, 1, 1)),
                0,
                0x123456789ABCDEF0,
            )
        ],
        acquire=False,
        conservative_dependencies=False,
        stateful=False,
    )

    assert words == (
        packet3(0x76, 3, compute=True),
        0x20C,
        0x100,
        0,
        packet3(0x76, 3, compute=True),
        0x212,
        0x11,
        0x22,
        packet3(0x76, 2, compute=True),
        0x228,
        0x33,
        packet3(0x76, 2, compute=True),
        0x218,
        0,
        packet3(0x76, 4, compute=True),
        0x207,
        256,
        1,
        1,
        packet3(0x76, 2, compute=True),
        0x215,
        0,
        packet3(0x76, 7, compute=True),
        0x240,
        0,
        0,
        0,
        0,
        0x9ABCDEF0,
        0x12345678,
        packet3(0x15, 4, compute=True),
        1,
        1,
        1,
        0x800D,
        *wait_compute_idle(),
    )


def test_conservative_graph_brackets_and_state_elision_are_exact() -> None:
    dispatch = (
        _image(),
        DispatchGeometry(grid_workitems=(256, 1, 1), block=(256, 1, 1)),
        0,
        0x100000,
    )
    stateless = encode_gfx1100_graph([dispatch, dispatch], stateful=False)
    stateful = encode_gfx1100_graph([dispatch, dispatch], stateful=True)
    local_cache = encode_gfx1100_graph(
        [dispatch, dispatch], stateful=True, local_cache_dependencies=True
    )

    assert stateless[: len(acquire_system())] == acquire_system()
    boundary_start = len(acquire_system()) + 35  # first dispatch plus final wait
    assert (
        stateless[boundary_start : boundary_start + len(dependency_global())] == dependency_global()
    )
    assert (
        local_cache[boundary_start : boundary_start + len(dependency_local_cache())]
        == dependency_local_cache()
    )
    assert dependency_global()[-1] == 0x0C380
    assert dependency_local_cache()[-1] == 0x00380
    assert len(local_cache) == len(stateful)
    assert len(stateful) < len(stateless)
    assert stateful[-2:] == wait_compute_idle()


def test_gfx1100_encoder_rejects_unsupported_abi_and_geometry() -> None:
    geometry = DispatchGeometry(grid_workitems=(256, 1, 1), block=(256, 1, 1))
    with pytest.raises(Pm4InspectionError, match="scratch"):
        encode_gfx1100_graph([(_image(private_segment_size=16), geometry, 0, 0x1000)])
    with pytest.raises(Pm4InspectionError, match="implicit SGPR"):
        encode_gfx1100_graph(
            [
                (
                    _image(kernel_code_properties=ENABLE_SGPR_KERNARG_SEGMENT_PTR | (1 << 2)),
                    geometry,
                    0,
                    0x1000,
                )
            ]
        )
    with pytest.raises(Pm4InspectionError, match="integral workgroups"):
        encode_gfx1100_graph(
            [
                (
                    _image(),
                    DispatchGeometry(grid_workitems=(257, 1, 1), block=(256, 1, 1)),
                    0,
                    0x1000,
                )
            ]
        )
    with pytest.raises(Pm4InspectionError, match="256-byte aligned"):
        encode_gfx1100_graph([(_image(code_entry=0x10040), geometry, 0, 0x1000)])


def test_vendor_pm4_ib_packet_bytes_are_exact() -> None:
    packet, publication = vendor_pm4_ib_packet(
        address=0x123456789ABCDEF0,
        dwords=0x345,
        completion_signal=0x0FEDCBA987654321,
    )

    assert len(packet) == 64
    assert publication == 0x00010100
    assert struct.unpack_from("<I", packet, 0)[0] == publication
    assert struct.unpack_from("<I", packet, 4)[0] == packet3(0x3F, 3, compute=False)
    assert struct.unpack_from("<I", packet, 8)[0] == 0x9ABCDEF0
    assert struct.unpack_from("<I", packet, 12)[0] == 0x12345678
    assert struct.unpack_from("<I", packet, 16)[0] == 0x30800345
    assert struct.unpack_from("<I", packet, 20)[0] == 10
    assert struct.unpack_from("<Q", packet, 56)[0] == 0x0FEDCBA987654321

    with pytest.raises(Pm4InspectionError, match="address"):
        vendor_pm4_ib_packet(address=3, dwords=1, completion_signal=1)
    with pytest.raises(Pm4InspectionError, match="dword"):
        vendor_pm4_ib_packet(address=0x1000, dwords=0, completion_signal=1)
