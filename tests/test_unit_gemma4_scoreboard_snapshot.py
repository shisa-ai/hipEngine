"""Command construction for the Gemma 4 scoreboard snapshot.

The snapshot advertises ``--gpu 0`` as the W7900 lane, but the campaign bench
validates ``--expect-gpu`` against the name of logical device 0 and defaults
to the XTX name - so a W7900 run died with "expected it to contain 'RX 7900
XTX'" instead of measuring. The expectation has to follow the selected lane.
"""

from __future__ import annotations

from scripts.gemma4_scoreboard_snapshot import expect_gpu_for


def test_expect_gpu_follows_the_selected_lane():
    assert expect_gpu_for(0) == "W7900"
    assert expect_gpu_for(1) == "RX 7900 XTX"
