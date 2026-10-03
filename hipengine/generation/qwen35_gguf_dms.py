"""Single-request scheduler adapter for compact DMS resident sessions."""
from hipengine.generation.qwen35_gguf import (
    Qwen35GGUFResidentModelRunner, _GGUFResidentSessionLease,
    _resident_session_wmma_prefill_default, hip_target_arch_environment,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.generation.deadline import raise_if_generation_deadline_expired


class Qwen35GGUFDMSModelRunner(Qwen35GGUFResidentModelRunner):
    def configure_engine_loop(self, config):
        if config.prefix_cache != "off":
            raise ValueError("DMS compact KV has no radix prefix-cache adapter; use prefix_cache='off'")
        if config.speculative_mtp_serving != "off":
            raise ValueError("DMS compact KV has no speculative serving adapter; use speculative_mtp_serving='off'")
        self._engine_loop_config = config
        self._prefix_cache_mode = "off"

    def _reserve_sessions(self):
        config = self.generator.dms_config
        session = Qwen35GGUFResidentSession(
            self.generator.model_path, backend=self.generator.backend,
            runtime=self._shared_runner.runtime, shared_runner=self._shared_runner,
            max_sequence_length=self._max_sequence_length, max_batch_size=1,
            use_wmma_prefill=_resident_session_wmma_prefill_default(),
            use_gemv_decode=True, dms_metadata_path=config.metadata_path,
            dms_prefill_mode=config.prefill_mode,
            dms_decision_mode=config.decision_mode,
            **self.generator._prepared_session_kv_kwargs(),
        )
        self._available.append(_GGUFResidentSessionLease(session, None))

    def _release_available_sessions(self):
        while self._available:
            self._available.pop().session.close()

    def reserve_admission(self, request):
        row = self._row(request.request_id)
        if not self._available:
            self._reserve_sessions()
        session = self._available[-1].session
        required = len(row.prompt_ids) + max(0, row.request.max_tokens - 1)
        if required > session.scratch.max_positions:
            raise ValueError(f"DMS request needs {required} positions, exceeding resident context {session.scratch.max_positions}")
        session.dms_max_new_tokens = max(1, int(row.request.max_tokens))

    def prefill_batch(self, work, *, commit):
        if not commit:
            raise ValueError("DMS prefill requires commit=True")
        self._begin_execution_step()
        for request_id, tokens in zip(work.request_ids, work.token_rows, strict=True):
            row = self._row(request_id)
            start = row.prefill_tokens_seen
            chunk = tuple(tokens)
            if chunk != row.prompt_ids[start:start + len(chunk)]:
                raise RuntimeError("DMS prefill token order changed")
            row.prefill_tokens_seen += len(chunk)
            if row.prefill_tokens_seen != len(row.prompt_ids):
                continue
            self._execution_step_request_id = int(request_id)
            self._mark_execution_stepped((request_id,))
            raise_if_generation_deadline_expired(row.request)
            with hip_target_arch_environment(self.generator.target_arch):
                if row.native_greedy:
                    self._prefill_native_row(row)
                elif row.native_sampled:
                    self._prefill_sampled_row(row)
                else:
                    self._run_resident_fallback(row)
            raise_if_generation_deadline_expired(row.request)

    def _step_native_c1_graph(self, row):
        # Compact extents and retention decisions change at each step; the
        # dense fixed-address graph ABI does not describe this store.
        return row.slot.session.step(int(row.slot.prev_token), return_logits=False)

    def _release_row_resources(self, row, *, retain_prefix_snapshots=False):
        super()._release_row_resources(row, retain_prefix_snapshots=False)
        # reset() resets dense recurrent state, not the compact retention owner.
        # Close the complete request session so no compact lease crosses requests.
        self._release_available_sessions()

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
