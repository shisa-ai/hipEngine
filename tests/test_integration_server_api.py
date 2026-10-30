"""Import-compatible home for helpers moved by the RF-12 server API split.

The tests formerly in this module now live in ``test_integration_server_*.py``
modules. This module holds no tests; it only keeps existing helper imports
such as ``from tests.test_integration_server_api import SpeculativeMTPFakeLLM``
working for external consumers.

Removal condition: delete this module and the legacy-import check in
``tests/test_integration_server_split_contract.py`` when
``rg -n "tests[.]test_integration_server_api" tests scripts`` matches only
those two files. Move any remaining importer to
``tests.support.server_api_fakes`` or ``tests.support.server_api_support``
first; never replace this shim with test re-exports, which would duplicate
collection.
"""

from tests.support.server_api_fakes import (
    AccountingSpeculativeMTPFakeLLM,
    AdmissionRejectedFakeLLM,
    ArtifactScopedSpeculativeMTPFakeLLM,
    BackendCancelledFakeLLM,
    BackendDeadlineFakeLLM,
    BackendErrorThenSequentialFakeLLM,
    BackendExecutionFailedFakeLLM,
    BlockingPrepareFakeLLM,
    C2OnlyArtifactScopedSpeculativeMTPFakeLLM,
    DeclaredCellSpeculativeMTPFakeLLM,
    DefaultMTPDeclaredCellFakeLLM,
    DelayedFakeLLM,
    DetailedGenerateFakeLLM,
    FailingSpeculativeMTPFakeLLM,
    FakeLLM,
    LateResolvingMTPFakeLLM,
    NarrowCapacityFakeLLM,
    NoTokenizerFakeLLM,
    OverDeclaredDepthSpeculativeMTPFakeLLM,
    PromotedArtifactScopedSpeculativeMTPFakeLLM,
    PromptPreparingFakeLLM,
    RealizedGroupArtifactScopedSpeculativeMTPFakeLLM,
    ResidentSessionPromptFakeLLM,
    SchedulerChunkRowsFakeLLM,
    ScratchProbeFailureFakeLLM,
    SequentialFakeLLM,
    SpeculativeMTPFakeLLM,
    WideScratchProbeFakeLLM,
    _FakeCapacityEstimate,
    _fake_kv_estimate,
)
from tests.support.server_api_support import (
    AR_BATCH_ENV,
    REPO_ROOT,
    _QWEN35_XML_TOOL_CALL,
    _WEATHER_TOOL,
    _api_error_taxonomy_table,
    _asgi_chat_post,
    _assert_openai_stream_tool_call_delta_shape,
    _assert_openai_tool_call_shape,
    _bounded_tool_schema,
    _continuation_capability,
    _fake_kv_pool_metadata,
    _fake_kv_pool_stats,
    _full_coverage_accounting,
    _incomplete_duplicate_read_tool_output,
    _labeled_metric_value,
    _load_single_replay_artifact,
    _metric_value,
    _mtp_accounting_detail,
    _parallelism_capability,
    _response_json_schema,
    _routing_metadata,
    _session_commit_policy_capability,
    _session_metadata_capability,
    _sse_payloads,
    _stateless_finish_details,
    _test_static_eligibility,
    _unified_diff_text,
)
