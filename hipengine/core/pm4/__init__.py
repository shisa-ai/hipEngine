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
from hipengine.core.pm4.native import (
    NativePm4Buffer,
    NativePm4Context,
    NativePm4Error,
    NativePm4Executable,
)
from hipengine.core.pm4.native_build import build_pm4_native, plan_pm4_native_build
from hipengine.core.pm4.packets import DispatchGeometry, Gfx1100KernelImage
from hipengine.core.pm4.transport import (
    DuplicateSubmissionTransportError,
    GraphSubmission,
    GraphSubmissionContext,
    GraphSubmissionRequest,
    HipGraphSubmission,
    MissingSubmissionTransportError,
    NativeGraphSubmission,
    NativeGraphSubmissionContext,
    SubmissionTransportKey,
    create_graph_submission,
    create_graph_submission_context,
    register_submission_transport,
    registered_submission_transports,
    resolve_submission_transport_factory,
    select_submission_transport,
)

__all__ = [
    "AmdgpuKernelMetadata",
    "HipDim3",
    "HipGraphManifest",
    "HipKernelNodeParams",
    "DispatchGeometry",
    "DuplicateSubmissionTransportError",
    "Gfx1100KernelImage",
    "GraphSubmission",
    "GraphSubmissionContext",
    "GraphSubmissionRequest",
    "HipGraphSubmission",
    "KernargField",
    "KernelNodeManifest",
    "LaunchContext",
    "MissingSubmissionTransportError",
    "NativeGraphSubmission",
    "NativeGraphSubmissionContext",
    "NativePm4Buffer",
    "NativePm4Context",
    "NativePm4Error",
    "NativePm4Executable",
    "Pm4InspectionError",
    "SelectedCodeObject",
    "SubmissionTransportKey",
    "build_pm4_native",
    "create_graph_submission",
    "create_graph_submission_context",
    "extract_elf_section",
    "inspect_hip_graph",
    "pack_kernargs",
    "parse_amdgpu_kernels",
    "parse_elf_sections",
    "plan_pm4_native_build",
    "register_submission_transport",
    "registered_submission_transports",
    "resolve_dso_for_function",
    "resolve_kernel_metadata",
    "resolve_submission_transport_factory",
    "select_amdgpu_code_object",
    "select_submission_transport",
    "topological_order",
]
