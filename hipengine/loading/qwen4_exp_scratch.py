"""Mandatory Qwen4Exp device scratch, excluding weights/KV/index/recurrent state.

These equations mirror the base runner allocation recipes and full-capacity
prefill repair queues. CPU census tests check them against the actual allocators.
Optional MMQ, graph, verification and transaction resources are not included.
"""


def qwen4_exp_scratch_breakdown(config, *, context_tokens: int, prefill_chunk_size: int):
    context, chunk = int(context_tokens), int(prefill_chunk_size)
    if not config.qsa_compression_ratio <= context <= config.context_length or chunk <= 0:
        raise ValueError("invalid Qwen4Exp scratch context/chunk")
    rows = min(context, chunk)
    h, branches, low = config.hidden_size, config.residual_branch_count, config.residual_low_rank
    ffn, experts, top_k = config.expert_feed_forward_length, config.expert_count, config.expert_used_count
    q_width = config.attention_head_count * config.attention_key_length
    kv_width = config.attention_kv_head_count * config.attention_key_length
    qkv_width = 2 * config.gdn_group_count * config.gdn_state_size + config.gdn_inner_size
    selected = config.qsa_dense_equivalent_max_tokens
    pages = (context + 255) // 256
    score_blocks = (context + config.qsa_compression_ratio - 1) // config.qsa_compression_ratio

    def gr(n):
        return 4 * n * (2 * branches * h + low + branches + h)

    def moe(n):
        compact = n * top_k
        tiles = (compact + 15 * experts + 15) // 16
        return (
            n * (4 * experts + 12 * h + 12 * ffn + 4)
            + compact * (40 + 2 * h + 10 * ffn)
            + 12 * experts + 16 * (experts + 1) + 8 * tiles + 8
        )

    def gdn_layer(n):
        mixer = 4 * n * (2 * qkv_width + 2 * config.gdn_inner_size
                         + 2 * config.gdn_time_step_rank + h)
        return 2 * gr(n) + mixer + moe(n) + n * (4 * branches * h + 4 * h)

    def qsa_layer(n):
        mixer = 4 * n * (6 * q_width + 3 * kv_width
                         + 2 * config.indexer_head_count * config.indexer_key_length
                         + config.indexer_key_length + h)
        return 2 * gr(n) + mixer + moe(n) + n * (4 * branches * h + 4 * h)

    def ple(n):
        return n * (22 * branches * h + 4 * h + 4 * branches)

    repair_indices = rows * top_k * max(h, 2 * ffn)
    if repair_indices > 2**31 - 1:
        raise ValueError("Qwen4Exp repair queue exceeds its int32 counter capacity")
    repair = 4 + 4 * repair_indices
    argmax_blocks = (config.vocab_size + 1023) // 1024
    return {
        "gdn_scratch": gdn_layer(1),
        "qsa_scratch": qsa_layer(1) + 12 * config.attention_head_count * selected,
        "ple_scratch": ple(1),
        "head_scratch": gr(1),
        "_buffers": 36 + 6 * h + 2 * branches * h + 4 * config.vocab_size + 12 * argmax_blocks,
        "attention_metadata": config.qsa_layer_count * (4 * pages + 16),
        "gdn_prefill_scratch": gdn_layer(rows) + repair,
        "qsa_prefill_scratch": qsa_layer(rows) + repair,
        "ple_prefill_scratch": ple(rows),
        "qsa_prefill_metadata": rows * (4 * pages + 20 + 8 * selected + 4 * score_blocks),
        "_prefill_buffers": rows * (32 + 6 * h + 2 * branches * h),
    }
