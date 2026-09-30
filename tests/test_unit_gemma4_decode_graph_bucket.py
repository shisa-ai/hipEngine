"""Decode-graph bucket arithmetic, without a GPU.

The session freezes launch shape to a context bucket; the bucket maths is the
part every capture depends on and the part a regression would silently break:
a bucket that does not cover the live position, one whose key-count range
straddles a route transition, or a frozen ``key_begin`` above the smallest
actual one would each break replay-vs-launched equality in a different way.

The route invariant under test: the global layers' tiled route admits
exactly at key counts that are multiples of ``KEY_TILE``, and the launched
path picks its route from the *live* key count (``position + 1``) while the
capture picks it from the bucket end. Those must agree at every position the
bucket serves.
"""

from __future__ import annotations

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import KEY_TILE
from hipengine.runtime.gemma4_decode_graph import BUCKET_WIDTH, _Bucket


def _route_is_tiled(key_count: int) -> bool:
    """The decode-time route for a global layer: tiled iff KEY_TILE divides."""

    return key_count % KEY_TILE == 0


def test_bucket_covers_the_live_position():
    for position in (0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 255, 256, 4095, 4096):
        bucket = _Bucket.for_position(position)
        assert bucket.start <= position < bucket.end
        assert bucket.end >= position + 1


def test_route_agrees_at_every_position_of_its_bucket():
    # The graph launches the tiled route iff KEY_TILE divides the bucket end;
    # the launched path does so iff KEY_TILE divides the live key count.
    # Every position in a bucket must see the same answer from both.
    for position in range(0, 3 * KEY_TILE + BUCKET_WIDTH):
        bucket = _Bucket.for_position(position)
        graph_tiled = _route_is_tiled(bucket.end)
        for served in range(bucket.start, bucket.end):
            launched_tiled = _route_is_tiled(served + 1)
            assert graph_tiled == launched_tiled, (
                f"position {position}'s bucket [{bucket.start}, {bucket.end}) "
                f"launches tiled={graph_tiled} but position {served} would be "
                f"launched tiled={launched_tiled}"
            )


def test_bucket_crossing_at_a_tile_boundary_is_a_singleton():
    # position KEY_TILE - 1 has live key count KEY_TILE: both paths must use
    # the tiled route with identical parameters, which only a bucket serving
    # exactly that key count can guarantee.
    bucket = _Bucket.for_position(KEY_TILE - 1)
    assert (bucket.start, bucket.end) == (KEY_TILE - 1, KEY_TILE)


def test_grid_bucket_stops_short_of_a_tile_boundary():
    # The grid bucket that would end at KEY_TILE stops one short, so its key
    # counts stay off the tiled admission point.
    bucket = _Bucket.for_position(KEY_TILE - 2)
    assert bucket.end == KEY_TILE - 1
    assert bucket.start <= KEY_TILE - 2
    assert bucket.end % KEY_TILE != 0


def test_buckets_advance_without_gaps():
    # Consecutive positions never skip a bucket start: the next bucket must
    # cover the next position and begin no later than it.
    previous = _Bucket.for_position(0)
    for position in range(1, 4 * KEY_TILE):
        bucket = _Bucket.for_position(position)
        assert bucket.start <= position
        if bucket.start != previous.start:
            assert bucket.start == position, (
                f"bucket at position {position} starts at {bucket.start}, "
                "leaving a gap or overlap"
            )
        previous = bucket


def test_bucket_width_is_not_derived_from_the_sequence():
    # The bucket depends only on the position, so a custom width moves the
    # grid boundary predictably away from tile boundaries.
    narrow = _Bucket.for_position(10, width=16)
    assert (narrow.start, narrow.end) == (0, 16)
    wide = _Bucket.for_position(10, width=32)
    assert (wide.start, wide.end) == (0, 32)