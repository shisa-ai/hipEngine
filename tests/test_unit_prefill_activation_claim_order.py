"""Claim-holding chunks must not admit another claim opener in the same item."""
from hipengine.generation.batch_scheduler import ResidentBatchScheduler


def test_gguf_independent_activations_share_chunked_prefill_item():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    scheduler = ResidentBatchScheduler(capacity=3, context_bucket_size=4)
    requests = [scheduler.submit(list(range(n)), max_new_tokens=1) for n in (9, 2, 6)]
    scheduler.admit_pending()
    work = scheduler.next_prefill_batch_work(
        chunk_size=4, max_rows=3, needs_activation=runner.prefill_needs_activation,
    )
    assert work.request_ids == tuple(requests)


def test_chunked_claim_holder_defers_short_claim_opener():
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    long = scheduler.submit(list(range(9)), max_new_tokens=1)
    short = scheduler.submit([11, 12], max_new_tokens=1)
    scheduler.admit_pending()
    work = scheduler.next_prefill_batch_work(
        chunk_size=4, max_rows=2, needs_activation=lambda _: True,
    )
    assert work.request_ids == (long,)
    assert scheduler.active_batch.requests[short].next_prompt_index == 0


def test_short_claim_releases_before_later_long_opener():
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    short = scheduler.submit([11, 12], max_new_tokens=1)
    long = scheduler.submit(list(range(9)), max_new_tokens=1)
    scheduler.admit_pending()
    work = scheduler.next_prefill_batch_work(
        chunk_size=4, max_rows=2, needs_activation=lambda _: True,
    )
    assert work.request_ids == (short, long)
