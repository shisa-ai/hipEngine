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

__all__ = [
    "AmdgpuKernelMetadata",
    "HipDim3",
    "HipGraphManifest",
    "HipKernelNodeParams",
    "KernargField",
    "KernelNodeManifest",
    "LaunchContext",
    "Pm4InspectionError",
    "SelectedCodeObject",
    "extract_elf_section",
    "inspect_hip_graph",
    "pack_kernargs",
    "parse_amdgpu_kernels",
    "parse_elf_sections",
    "resolve_dso_for_function",
    "resolve_kernel_metadata",
    "select_amdgpu_code_object",
    "topological_order",
]
