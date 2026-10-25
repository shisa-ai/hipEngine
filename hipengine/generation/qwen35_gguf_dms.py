"""Typed DMS serving restrictions over the shared Qwen compact lifecycle."""
from hipengine.generation.qwen35_gguf import (
    Qwen35GGUFResidentModelRunner, _GGUFResidentSessionLease,
    _resident_session_wmma_prefill_default,
)


class Qwen35GGUFDMSModelRunner(Qwen35GGUFResidentModelRunner):
    def configure_engine_loop(self, config):
        if config.prefix_cache != "off":
            raise ValueError("DMS compact KV has no radix prefix-cache adapter; use prefix_cache='off'")
        if config.speculative_mtp_serving != "off":
            raise ValueError("DMS compact KV has no speculative serving adapter; use speculative_mtp_serving='off'")
        super().configure_engine_loop(config)

    def _reserve_sessions(self, *, max_sequence_length=None):
        # Shared acquisition owns policy binding, auto context, allocation
        # backoff and profile configuration. Compact leases are never returned
        # to this pool: the common request retirement path closes them.
        session, pool_key, _reused = self.generator._acquire_shared_session(
            self._shared_runner, pool_name="compact_dms_request",
            max_sequence_length=(self._max_sequence_length if max_sequence_length is None
                                 else max_sequence_length),
            max_batch_size=1, defer_kv_allocation=False,
            use_wmma_prefill=_resident_session_wmma_prefill_default(),
            use_gemv_decode=True,
        )
        self._available.append(_GGUFResidentSessionLease(session, pool_key))

    def _release_available_sessions(self):
        while self._available:
            self._available.pop().session.close()

    def observability_snapshot(self):
        result = super().observability_snapshot()
        session = self._session
        backend = None if session is None else getattr(session, "_dms_backend", None)
        result["retention"] = {"policy": "dms", "storage": "bf16",
                               "prefill_mode": self.generator.dms_config.prefill_mode,
                               "compact": None if backend is None else backend.observability_snapshot()}
        return result

    def _execution_metadata(self, row):
        result = super()._execution_metadata(row)
        result["retention"] = {"policy": "dms", "storage": "bf16",
                               "prefill_mode": self.generator.dms_config.prefill_mode,
                               "decision_mode": self.generator.dms_config.decision_mode}
        return result
