"""Strict, torch-free frontend for the in-tree ROCr/PM4 transport.

Importing this package only defines parsers and value objects. It does not load
HIP/HSA libraries, compile native code, or touch a GPU.
"""

from hipengine.core.hip import HipDim3, HipKernelNodeParams
from hipengine.core.pm4.elf import (
    SelectedCodeObject,
    extract_elf_section,
    parse_elf_sections,
    select_amdgpu_code_object,
)
from hipengine.core.pm4.errors import Pm4InspectionError
from hipengine.core.pm4.graph import (
    HipGraphManifest,
    KernelNodeManifest,
    inspect_hip_graph,
    resolve_dso_for_function,
    topological_order,
)
from hipengine.core.pm4.kernarg import LaunchContext, pack_kernargs
from hipengine.core.pm4.metadata import (
    AmdgpuKernelMetadata,
    KernargField,
    parse_amdgpu_kernels,
    resolve_kernel_metadata,
)
from hipengine.core.pm4.native import NativePm4Context, NativePm4Error, NativePm4Executable
from hipengine.core.pm4.native_build import build_pm4_native, plan_pm4_native_build
from hipengine.core.pm4.packets import DispatchGeometry, Gfx1100KernelImage

__all__ = [
    "AmdgpuKernelMetadata",
    "HipDim3",
    "HipGraphManifest",
    "HipKernelNodeParams",
    "DispatchGeometry",
    "Gfx1100KernelImage",
    "KernargField",
    "KernelNodeManifest",
    "LaunchContext",
    "NativePm4Context",
    "NativePm4Error",
    "NativePm4Executable",
    "Pm4InspectionError",
    "SelectedCodeObject",
    "build_pm4_native",
    "extract_elf_section",
    "inspect_hip_graph",
    "pack_kernargs",
    "parse_amdgpu_kernels",
    "parse_elf_sections",
    "plan_pm4_native_build",
    "resolve_dso_for_function",
    "resolve_kernel_metadata",
    "select_amdgpu_code_object",
    "topological_order",
]
