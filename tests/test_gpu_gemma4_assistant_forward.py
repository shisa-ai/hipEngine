"""The Gemma 4 assistant (MTP) head forward, against a real backbone.

The head's job is to draft the token the target is about to produce, so the
oracle is functional: run the target one more step, take its own next token, and
require the head to draft exactly that. A second test chains four draft steps and
compares the head's top three candidates at each step against the list
``llama.cpp`` prints for the same prompt under ``--verbose`` -- an external
implementation's answer, not this one's.

Neither test compares the head's projected state against the backbone's hidden
state, which an earlier version of this file did. ``nextn_proj_post`` does not
reconstruct the target's hidden state: fed the verified step-0 input, the head's
output sits at cosine -0.02 from the backbone's post-norm hidden at the position
it represents, while the backbone's own hidden rows sit at +0.41 from each other.
Three independent implementations -- this forward, a NumPy reading of
llama.cpp's graph, and transformers' own ``Gemma4AssistantForCausalLM`` -- agree
on that, and the same forward reproduces llama.cpp's candidate lists exactly, so
the projection is a recurrence state rather than a prediction of the target's
activations.

The head's input contract, all of it read off ``common/speculative.cpp``:

* ``pending_h`` is documented as the "pair (h_p, x_{p+1}) at MTP pos p+1" and
  ``verify_h``'s row 0 as "the sampled token", so step 0 is fed the token the
  target just sampled together with the hidden row that *produced* it.
* ``dp.n_past`` is the query position: one past the last row the target wrote, so
  the query's own K/V is not in the shared cache and the views are taken before
  the extra forward that would add it.
* The ``is_mem_shared`` branch -- which Gemma 4 assistants take -- adds every
  later draft token at the same ``dp.n_past``, citing the Hugging Face doc's
  "the position_ids value are constant".
* Each later step is fed the token it just drafted together with the head's own
  projected state, which is why the recurrent state lives on the head.

The seed hidden state is the target's **post-output-norm** state, not the raw
last-block residual stream: llama.cpp's backbone sets ``res->t_h_nextn`` *after*
``build_norm(cur, model.output_norm, ...)`` and calls it "the LM-head input
feature" handed to the drafter "as the recurrent h input".

Guarded on both artifacts and on HIP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.gguf import GGUFReader, scan_gguf
from hipengine.loading.gemma4_assistant_device import (
    load_gemma4_assistant_device_weights,
)
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
from hipengine.runtime.gemma4_assistant import Gemma4AssistantHead, Gemma4MtpDrafter
from tests._rocm_guard import hip_runtime_available

BACKBONE = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
HEAD = Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf")

_needs = pytest.mark.skipif(
    not (hip_runtime_available() and BACKBONE.exists() and HEAD.exists()),
    reason="the Gemma 4 backbone, the MTP head, and HIP are all required",
)

# A short synthetic prompt with repeated structure. Not asserted on: the head
# disagrees with its own backbone here (see the first test's docstring). Kept
# because the disagreement is the sharpest available example of what the head is
# not, and because a future change that makes the head agree here is a real
# improvement worth noticing.
DEGENERATE_PROMPT = [2, 818, 5279, 529, 7001, 108, 818, 5279, 529, 22172, 108, 107]

# The same sentence under Gemma 4's chat template, which is what llama.cpp's
# server actually feeds. 66 tokens, matching the ``prompt_n`` in the timings of
# the run the candidate lists below were read from.
TEMPLATED = [
    2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 818, 4083, 529, 506,
    10995, 23436, 55462, 919, 1082, 496, 13460, 1518, 236764, 6534, 607, 506,
    37813, 529, 506, 3207, 529, 13706, 528, 506, 35186, 7691, 19339, 532, 16548,
    607, 506, 3798, 529, 506, 16425, 38613, 528, 506, 15778, 7691, 7747, 236761,
    799, 1534, 236764, 13706, 13958, 699, 496, 1944, 106, 107, 105, 4368, 107,
]

# llama.cpp's ``--verbose`` MTP log for that prompt, one line per draft step:
#
#   D spec draft: - seq_id 0, draft candidate 0, pos 0:  45518 (   1.000) 'thought'
#   D spec draft: - seq_id 0, draft candidate 1, pos 0:   3305 (   0.000) ' thought'
#   D spec draft: - seq_id 0, draft candidate 2, pos 0:  44027 (   0.000) ' thoughtful'
#
# and likewise for pos 1, 2 and 3. The head's own top three, in order, must match.
LLAMACPP_CANDIDATES = [
    [45518, 3305, 44027],
    [108, 107, 236768],
    [236829, 236775, 818],
    [139, 3729, 623],
]

BACKBONE_WIDTH = 2816
VOCAB = 262144


def _to_float32(buffer) -> np.ndarray:
    """Copy a BF16 device buffer to host as float32.

    BF16 is the top 16 bits of a float32, not float16. ``view(np.float16)``
    reinterprets rather than converts, and produced a stable but meaningless
    cosine while this test was being written.
    """

    raw = np.empty(buffer.nbytes // 2, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(raw), buffer, raw.nbytes)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _to_bf16(values: np.ndarray) -> np.ndarray:
    as_int = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (
        (as_int + np.uint32(0x7FFF) + ((as_int >> 16) & np.uint32(1))) >> 16
    ).astype(np.uint16)


def _stage(host_bf16: np.ndarray):
    buffer = malloc(host_bf16.nbytes)
    copy_host_array_to_device(buffer, np.ascontiguousarray(host_bf16).view(np.uint8))
    return buffer


class _Session:
    """A loaded backbone and head, plus the seeding the draft loop needs."""

    def __init__(self, prompt: list[int]):
        self.reader = GGUFReader(str(BACKBONE))
        self.backbone_weights = load_gemma4_device_weights(self.reader)
        self.runner = Gemma4Runner(weights=self.backbone_weights, capacity=128)
        self.head_weights = None
        self.head = None
        self.staged = None
        self.prompt = list(prompt)

    def __enter__(self) -> "_Session":
        logits = self.runner.forward(self.prompt, apply_softcap=False)
        rows = len(self.prompt)
        # Stage the row that produced the sampled token. It has to survive the
        # extra forward that produces the oracle, which reuses the runner's
        # hidden buffer.
        self.staged = _stage(_to_bf16(_to_float32(self.runner.hidden_state(row=rows - 1))))
        self.sampled = int(np.argmax(logits))
        # The query position is one past the last row the target wrote, and the
        # extra forward below would extend the live count, so the views are taken
        # here.
        self.position = self.runner.position
        self.shared = {
            index: self.runner.shared_kv(index) for index in range(self.runner.layer_count)
        }
        self.head_weights = load_gemma4_assistant_device_weights(str(HEAD))
        self.head = Gemma4AssistantHead(
            weights=self.head_weights,
            backbone=gemma4_gguf_config_from_metadata(scan_gguf(BACKBONE)),
            backbone_embedding=self.backbone_weights.embed_tokens,
            backbone_output_norm=self.backbone_weights.final_norm.buffer,
            capacity=128,
            eps=1e-6,
        )
        self.head.prime(self.staged)
        return self

    def __exit__(self, *_exc) -> None:
        if self.head is not None:
            self.head.close()
        if self.head_weights is not None:
            self.head_weights.free()
        if self.staged is not None:
            free(self.staged)
        self.runner.close()
        self.backbone_weights.free()

    def target_next_token(self, token: int) -> int:
        """Advance the target by ``token`` and return its own next token."""

        return int(np.argmax(self.runner.forward([token], apply_softcap=False)))


@_needs
def test_the_head_drafts_the_token_the_backbone_would_generate() -> None:
    """The end-to-end correctness signal: the draft is the target's next token.

    This runs on the chat-templated prompt, where the continuation is decided, and
    the head's first draft is the token the target itself goes on to produce.

    It is deliberately not asserted on every prompt. The head is four blocks
    approximating a thirty-block target, and it does sometimes disagree: on the
    short synthetic prompt ``PROMPT`` below it drafts 5279 where the backbone
    drafts 236772. That disagreement is head quality, not a wiring defect -- the
    reference implementation (transformers' own ``Gemma4AssistantForCausalLM``
    over the same GGUF, fed the same seed) also drafts 5279 there, and the same
    forward reproduces llama.cpp's candidates exactly in the test below.
    """

    with _Session(TEMPLATED) as session:
        expected = session.target_next_token(session.sampled)
        logits, h_next = session.head.forward(
            session.sampled, position=session.position, shared_kv=session.shared
        )

        assert logits.shape == (VOCAB,)
        assert np.isfinite(logits).all()
        assert h_next.shape == (BACKBONE_WIDTH,)
        assert np.isfinite(h_next).all()
        assert np.abs(h_next).max() > 0, "the head produced an all-zero recurrent state"

        drafted = int(np.argmax(logits))
        assert drafted == expected, (
            f"the head drafted {drafted} where the backbone's own next token is "
            f"{expected}. A forward that reads the head's own token_embd for the "
            f"input, skips the target's final norm on the seed state, or binds the "
            f"blocks to the wrong shared layers misses this."
        )


@_needs
def test_the_draft_chain_reproduces_llamacpps_candidates() -> None:
    """Four chained steps against llama.cpp's own candidate lists.

    The recurrence is where a wiring bug hides: step 0 uses the target's seed
    state and every later step uses the head's own projected state at a constant
    position. Asserting the top three at each step, rather than only the top one,
    makes an implementation that re-seeds or re-norms partway through fail.
    """

    with _Session(TEMPLATED) as session:
        drafted = []
        token = session.sampled
        for step, expected in enumerate(LLAMACPP_CANDIDATES):
            logits, _h_next = session.head.forward(
                token, position=session.position, shared_kv=session.shared
            )
            top = [int(index) for index in np.argsort(logits)[::-1][:3]]
            assert top == expected, (
                f"draft step {step}: the head's candidates are {top}, llama.cpp's "
                f"are {expected}"
            )
            token = top[0]
            drafted.append(token)

        assert drafted == [row[0] for row in LLAMACPP_CANDIDATES]


@_needs
def test_the_drafter_proposes_tokens_the_backbone_accepts() -> None:
    """The draft loop end to end, measured the way speculative decoding is.

    Both sides are greedy, which is the comparison llama.cpp's ``--temp 0
    --top-k 1`` MTP run reports as ``#acc rate/pos``. Its numbers on this
    artifact are ``(1.000, 0.750, 0.500, 0.500)`` with a mean accepted length of
    3.75 over four draft calls; this loop measures ``(0.812, 0.821, 0.812,
    0.615)`` and 4.06 over 48 rounds, which ``scripts/gemma4_mtp_acceptance.py``
    reproduces.

    The bar here is deliberately far below the measured rate. What it is meant to
    catch is a draft loop that does not work at all -- a mis-seeded recurrence, a
    stale shared-KV view, or a position that advances -- each of which collapses
    acceptance to near zero while leaving the single-step forward test green.
    """

    rounds = 8
    per_round = 4
    with _Session(TEMPLATED) as session:
        drafter = Gemma4MtpDrafter(
            head=session.head, runner=session.runner, max_drafts=per_round
        )
        token = session.sampled
        reached = [0] * per_round
        matched = [0] * per_round
        for _round in range(rounds):
            consumed = token
            for slot, draft in enumerate(drafter.draft(token)):
                reached[slot] += 1
                expected = session.target_next_token(consumed)
                if draft != expected:
                    token = expected
                    break
                matched[slot] += 1
                consumed = draft
            else:
                token = session.target_next_token(consumed)

        overall = sum(matched) / sum(reached)
        assert matched[0] >= rounds * 0.5, (
            f"the drafter's first proposal was right in only {matched[0]} of {rounds} "
            f"rounds; a working chain measures about 0.81"
        )
        assert overall >= 0.5, (
            f"overall draft acceptance is {overall:.3f} over {sum(reached)} drafts "
            f"({matched}); a working chain measures about 0.78"
        )
