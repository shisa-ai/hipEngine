"""Shared geometry fixtures for the Qwen4Exp batch-runner tests.

The batch runner composes committed per-stage owners, so its tests need one
model geometry that every stage can address: ``hidden`` and ``ffn`` are
multiples of the Q4_K block so the MoE experts can be real quantised weights,
the index compression ratio divides the token budget, and the PLE row width
times the n-gram head count equals ``hidden``.

Kept out of ``test_*`` modules so both the mocked unit tier and the GPU tier can
import it (``docs/TESTING.md``, "Test Naming and Discovery").
"""

from __future__ import annotations

import numpy as np

from hipengine.loading.qwen4_exp_gguf import Qwen4ExpGGUFConfig

HIDDEN = 256
VOCAB = 64
BRANCHES = 2
LOW_RANK = 32
RESIDUAL_WIDTH = BRANCHES * HIDDEN
BLOCK_COUNT = 3
LAYER_TYPES = ("gdn", "qsa", "gdn")
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIM = 8
Q_WIDTH = QUERY_HEADS * HEAD_DIM
KV_WIDTH = KV_HEADS * HEAD_DIM
ROTARY_DIM = 8
INDEX_HEADS = 2
INDEX_DIM = 8
RATIO = 2
TOKEN_BUDGET = 4
BLOCK_BUDGET = TOKEN_BUDGET // RATIO
OUTPUT_STRIDE = BLOCK_BUDGET * RATIO + RATIO - 1
DENSE_EQUIVALENT_LIMIT = TOKEN_BUDGET + RATIO - 1
GDN_CONV_KERNEL = 4
GDN_K_HEADS = 2
GDN_V_HEADS = 4
GDN_HEAD_DIM = 128  # the indexed GDN decode ABI constant
GDN_INNER = GDN_V_HEADS * GDN_HEAD_DIM
GDN_QKV_WIDTH = 2 * GDN_K_HEADS * GDN_HEAD_DIM + GDN_V_HEADS * GDN_HEAD_DIM
GDN_CORE_WIDTH = GDN_V_HEADS * GDN_HEAD_DIM
EXPERTS = 4
TOP_K = 2
FFN = 256
PLE_LAYERS = (0,)
PLE_NGRAM = 3
PLE_HEADS_PER_NGRAM = 16
PLE_HEAD_COUNT = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM
PLE_ROW_WIDTH = HIDDEN // PLE_HEAD_COUNT
PLE_CONV_KERNEL = 4
PLE_EOS = VOCAB - 1
PLE_IMAGE = VOCAB - 2
PLE_HEAD_VOCAB_SIZES = tuple(3 for _ in range(PLE_HEAD_COUNT))
PLE_HEAD_OFFSETS = tuple(index * 3 for index in range(PLE_HEAD_COUNT))
PLE_ROW_COUNT = sum(PLE_HEAD_VOCAB_SIZES)
PLE_LAYER_MULTIPLIERS = (3, 5, 7)
CONTEXT_LENGTH = 4096
MAX_POSITIONS = 300
BLOCK_SIZE = 256


def config() -> Qwen4ExpGGUFConfig:
    """The one synthetic geometry every batch-runner test uses."""

    return Qwen4ExpGGUFConfig(
        architecture="qwen4exp",
        block_count=BLOCK_COUNT,
        context_length=CONTEXT_LENGTH,
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        residual_branch_count=BRANCHES,
        residual_low_rank=LOW_RANK,
        full_attention_interval=2,
        attention_compress_ratios=(0, RATIO, 0),
        layer_types=LAYER_TYPES,
        attention_head_count=QUERY_HEADS,
        attention_kv_head_count=KV_HEADS,
        attention_key_length=HEAD_DIM,
        attention_value_length=HEAD_DIM,
        attention_rms_epsilon=1e-6,
        indexer_head_count=INDEX_HEADS,
        indexer_key_length=INDEX_DIM,
        qsa_token_budget=TOKEN_BUDGET,
        rope_dimension_count=ROTARY_DIM,
        rope_dimension_sections=(3, 3, 2, 0),
        rope_freq_base=10_000.0,
        gdn_conv_kernel=GDN_CONV_KERNEL,
        gdn_group_count=GDN_K_HEADS,
        gdn_inner_size=GDN_INNER,
        gdn_state_size=GDN_HEAD_DIM,
        gdn_time_step_rank=GDN_V_HEADS,
        expert_count=EXPERTS,
        expert_used_count=TOP_K,
        expert_feed_forward_length=FFN,
        shared_expert_feed_forward_length=FFN,
        ple_layers=PLE_LAYERS,
        ple_ngram_size=PLE_NGRAM,
        ple_heads_per_ngram=PLE_HEADS_PER_NGRAM,
        ple_conv_kernel=PLE_CONV_KERNEL,
        ple_eos_token_id=PLE_EOS,
        ple_image_token_id=PLE_IMAGE,
        ple_row_width=PLE_ROW_WIDTH,
        ple_layer_multipliers=PLE_LAYER_MULTIPLIERS,
        ple_head_offsets=PLE_HEAD_OFFSETS,
        ple_head_vocab_sizes=PLE_HEAD_VOCAB_SIZES,
        tokenizer_bos_token_id=0,
        tokenizer_eos_token_id=1,
        tokenizer_padding_token_id=0,
    )


class FakePleTable:
    """Deterministic PLE row table with the mmap table's read interface."""

    def __init__(self) -> None:
        rows = np.arange(PLE_ROW_COUNT * PLE_ROW_WIDTH, dtype=np.float32)
        self.values = (rows.reshape(PLE_ROW_COUNT, PLE_ROW_WIDTH) * 1e-3).astype(
            np.float32
        )
        self.row_width = PLE_ROW_WIDTH
        self._telemetry = None

    def gather_rows(self, indices: np.ndarray) -> np.ndarray:
        return self.values[np.asarray(indices, dtype=np.int64)]

    def telemetry(self) -> None:
        return None
