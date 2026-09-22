"""Types shared across the Gemma 4 gfx1100 kernel family.

``Gemma4Projection`` lives here rather than in :mod:`gemma4_layer` because both
the layer and the expert forward annotate their weight arguments with it, and the
layer already imports from the expert module. Defining it in either one would
make the import graph circular.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hipengine.loading.gemma4_gguf_device import Gemma4GGUFDeviceWeight

    # A projection weight is a bf16 device pointer (``int``) or a GGUF device
    # weight carrying quantized blocks. Both are resident layouts a Gemma 4
    # artifact can legitimately use, so the dispatch is on the value, not on a
    # flag.
    Gemma4Projection = int | Gemma4GGUFDeviceWeight
else:
    Gemma4Projection = object

__all__ = ["Gemma4Projection"]
