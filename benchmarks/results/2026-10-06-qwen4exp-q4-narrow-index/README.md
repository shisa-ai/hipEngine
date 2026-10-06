# EG-5 checked Q4_K gate/up index screen

This diagnostic screen tested a checked-int32 device-index sibling of the
three-plane, 16-row Q4_K iu8-risk gate/up owner. The public launcher and all
map buffers stayed int64. The sibling narrowed only checked device
address/control intermediates after complete byte-range admission checks.
Production remains on the wide parent.

## Result

On gfx1151/Radeon 8060S host `55ea6c509d0b49eea8de7094a1023668`, the complete
natural operation included count, prefix, scatter/gather, tile map, risk
collection, sparse exact repair and SiLU. It covered code, English, Japanese
and mixed Japanese/English inputs at 512, 1024 and 4096 prompt tokens, with
layers 0, 24 and 47. The sibling was slower in all 18 case/layer medians:
**+0.089% to +0.474%**, aggregate median **+0.276%**. It is not promoted.

Outputs after repair, SiLU outputs, risk membership, input/weight bytes and
teardown matched the parent. The focused suite covered 26 CPU contracts and
21 GPU cases, including unequal output planes, empty and tail experts,
nonfinite/subnormal/large inputs, queue capacities, invalid maps, a nondefault
stream and an independent CPU-reference floor.

The cached code-object comparison found 207 vs 211 VGPR, 0 private bytes and
4,336 vs 4,388 static instructions. Carry pairs fell from 69 to 39. Both
variants remained in the same register-allocation bucket and scratch-free.
A cached `rocprofv3 --kernel-trace` operation trace confirms that both the
wide and checked symbols launched with plausible durations. The symbol rows
had different launch counts because the diagnostic hook is layered into a live
model run, so those per-symbol durations are not an independent A/B result;
the complete replay above is the timing evidence.

## Reproduction

See `screen.json` for source identity, commands, resource fields and artifact
paths. The raw natural-operation JSON, code objects and trace remain under
`/tmp/qwen4exp-q4-narrow-20261006/` and are not committed.
