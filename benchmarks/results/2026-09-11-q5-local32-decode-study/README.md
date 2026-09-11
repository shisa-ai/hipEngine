# Q5 dense local32 decode study + admission (2026-09-11, decode lever 3)

## The study

- trace-unrouted.csv: rocprofv3 kernel trace before the route. The Q5
  direct GEMV (`qk_t16_selected_direct_gemv_kernel<u16,5,...>`) runs
  73.7 us/call with VGPR 128, LDS 512 B, 128-thread blocks; the Q4
  local32 comparator runs 45.3 us/call with VGPR 96, LDS 0, 32-thread
  blocks. The direct kernel re-reads and re-decodes the per-column
  d/dmin and the superblock scale/min bytes inside the K loop - four
  redundant memory ops plus two fp16 decodes per MAC.
- microbench.py: synthetic T16 tiles at the production shapes, direct
  vs local32. GPU1 (XTX): 2.3-2.6x (91.8->39.6 us at 5120x6144,
  191.5->77.5 at 17408x5120). GPU0: 1.6-2.1x. The T16 layout needs no
  repack: the QL nibble window is the same aligned u32 per lane-row the
  Q4 owner reads; the Q5 high bit adds one u8 per lane-row.
- trace-routed.csv: after the route, the local32 kernel serves every Q5
  decode single (3036 calls, 53.5 us/call in-situ); the direct kernel
  is gone from the decode window.

## Admission

- probes/: scripts/gguf_q5_local32_decode_gate.py (the C1-table arm
  swap; the IQ-policy probe cannot gate this route). Natural
  self-generated prompts + the 18 tokenized category fixtures, pooled
  campaign predicate, 3 seeds x both artifacts: ALL PASS.
  K_M mean 8.9e-5, max 2.70e-2, top-1 99.92%; K_S mean 6.7e-5,
  max 8.1e-3, top-1 99.92%. The category half is seed-independent
  (fixed fixtures, incumbent-greedy extension), so per-seed runs are
  reruns for that half; the natural half varies per seed.
- sweeps/: decode (GPU1 XTX, 512/128, median of 3): K_M 27.70 ->
  29.98 tok/s (+8.2%), K_S 28.41 -> 29.49. Prefill unchanged.
