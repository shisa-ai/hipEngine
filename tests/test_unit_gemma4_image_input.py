"""CPU checks for Gemma image spans at the text-backbone boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.gemma4 import _keep_mask, gemma4_image_block_ranges


def test_image_mask_is_bidirectional_only_inside_sliding_image_block():
    local = SimpleNamespace(sliding_window=3)
    global_ = SimpleNamespace(sliding_window=None)
    spans = ((2, 5), (6, 8))
    mask = _keep_mask(local, 0, 9, image_spans=spans)
    assert mask[2, 4] == 1
    assert mask[2, 5] == 0
    assert mask[4, 1] == 0
    assert mask[6, 7] == 1
    assert mask[4, 6] == 0
    assert mask[1, 2] == 0
    np.testing.assert_array_equal(
        _keep_mask(global_, 0, 9, image_spans=spans),
        np.tril(np.ones((9, 9), dtype=np.uint8)),
    )


def test_image_block_ranges_never_split_a_bidirectional_span():
    assert gemma4_image_block_ranges(18, 8, ((6, 11),)) == ((0, 6), (6, 14), (14, 18))
    assert gemma4_image_block_ranges(17, 8, ((1, 4), (9, 16))) == ((0, 8), (8, 16), (16, 17))
    assert gemma4_image_block_ranges(2, 8, ()) == ((0, 2),)
    with pytest.raises(ValueError, match="max_block"):
        gemma4_image_block_ranges(14, 4, ((1, 6),))


def test_image_mask_does_not_change_subsequent_decode_or_global_attention():
    attn = SimpleNamespace(sliding_window=4)
    spans = ((1, 4),)
    np.testing.assert_array_equal(
        _keep_mask(attn, 7, 1, image_spans=spans),
        _keep_mask(attn, 7, 1),
    )
    assert _keep_mask(attn, 2, 2, image_spans=spans)[0, 3] == 1


class ImageTokenizer:
    token_to_id = {'<|image>': 10, '<|image|>': 11, '<image|>': 12}

    def encode(self, text, *, add_special_tokens=False):
        import re
        return [self.token_to_id.get(piece, 20) for piece in re.split(
            r'(<\|image\|>|<\|image>|<image\|>)',text,
        ) if piece]


def test_prompt_expands_ordered_images_and_preserves_preformatted_thinking():
    from hipengine.generation.gemma4_multimodal import gemma4_image_prompt, gemma4_embedding_overrides
    tokenizer = ImageTokenizer()
    prompt = '<|turn>user\nfirst <|image|> then <|image|><turn|>\n<|turn>model\n'
    ids, spans = gemma4_image_prompt(prompt,(2,3),tokenizer,lambda _: pytest.fail('rendered twice'))
    assert spans == ((2,4),(7,10))
    assert len(ids) == 12
    features = (np.ones((2,5)), np.full((3,5), 7.0))
    overrides = gemma4_embedding_overrides(features,spans)
    assert tuple(overrides) == (2,3,7,8,9)
    np.testing.assert_array_equal(overrides[7], features[1][0])
    assert overrides[2].dtype == np.float32


def test_plain_image_prompt_uses_chat_template_and_missing_marker_is_inserted():
    from hipengine.generation.gemma4_multimodal import gemma4_image_prompt
    calls = []
    def render(messages):
        calls.append(messages)
        return '<|turn>user\n'+messages[0]['content']+'<turn|>\n<|turn>model\n'
    _,spans = gemma4_image_prompt('Describe this.',(4,),ImageTokenizer(),render)
    assert spans == ((2,6),)
    assert calls == [[{'role':'user','content':'<|image|>\nDescribe this.'}]]
    with pytest.raises(ValueError,match='markers'):
        gemma4_image_prompt('<|image|>',(2,3),ImageTokenizer(),render)


def test_media_preserves_order_and_refuses_unsupported_modalities():
    from hipengine.generation.gemma4_multimodal import gemma4_media_images
    a,b = np.zeros((8,12,3),np.uint8),np.ones((12,8,3),np.uint8)
    assert gemma4_media_images(a)[0] is a
    images = gemma4_media_images({'items':[{'type':'image','data':a},{'type':'image','data':b}]})
    assert images[0] is a and images[1] is b
    with pytest.raises(NotImplementedError,match='video or audio'):
        gemma4_media_images({'items':[{'type':'video','data':a}]})
    with pytest.raises(ValueError,match='at least one'):
        gemma4_media_images({'items':[]})


def test_image_scope_clears_on_exception_and_does_not_scale_projected_rows():
    from hipengine.runtime.gemma4 import Gemma4Runner, _bf16_bits
    runner = object.__new__(Gemma4Runner)
    runner.weights = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    runner.capacity,runner.max_block = 32,8
    row = np.array([1.5,2.5,3.5,4.5],np.float32)
    with pytest.raises(RuntimeError,match='simulated'):
        with runner.image_input({3:row},((3,4),)):
            np.testing.assert_array_equal(runner._image_embeddings[3],_bf16_bits(row))
            with pytest.raises(RuntimeError,match='already active'):
                with runner.image_input({3:row},((3,4),)):
                    pass
            raise RuntimeError('simulated failure')
    assert runner._image_embeddings is None and runner._image_spans == ()


def test_generator_routes_projected_rows_and_reports_expanded_usage(monkeypatch):
    from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
    from hipengine.generation.registry import GenerationRequest, GenerationOutput
    from hipengine.runtime.gemma4 import Gemma4Runner
    generator = Gemma4GGUFGenerator('/unused.gguf',None,None,context_length=64)
    generator._tokenizer = ImageTokenizer()
    generator._vision_config = SimpleNamespace(
        patch_size=2,pooling_kernel_size=2,max_soft_tokens=4,check_grid=lambda *grid: None,
    )
    generator._vision_runner = SimpleNamespace(forward=lambda patches,positions,grid,request:
                                               np.ones((4,4),np.float32))
    runner = object.__new__(Gemma4Runner)
    runner.weights = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    runner.capacity,runner.max_block = 64,16
    monkeypatch.setattr(generator,'_ensure_runner',lambda **kwargs: runner)
    monkeypatch.setattr(generator,'render_chat_prompt',lambda messages: '<|turn>'+messages[0]['content'])
    def decode(request):
        assert len(request.prompts) == 1 and not isinstance(request.prompts[0],str)
        assert runner._image_spans == ((2,6),)
        assert set(runner._image_embeddings) == {2,3,4,5}
        return (GenerationOutput(text='green',generated_token_ids=(20,)),)
    monkeypatch.setattr(generator,'generate_detailed',decode)
    assert generator.count_multimodal_tokens('color?',np.zeros((8,8,3),np.uint8)) == 8
    out = generator.generate_multimodal_detailed('color?',np.zeros((8,8,3),np.uint8),
                                                GenerationRequest(prompts=('color?',),max_tokens=3,temperature=0,top_p=1,ignore_eos=False))
    assert out.text == 'green' and out.prompt_tokens == 8
    assert runner._image_embeddings is None
    with pytest.raises(NotImplementedError,match='INT8 attention has no bidirectional'):
        generator.generate_multimodal_detailed('color?',None,
            GenerationRequest(prompts=('color?',),max_tokens=1,temperature=0,top_p=1,ignore_eos=False,
                              kv_storage='int8_per_token_head'))


def test_image_attention_call_disables_causal_walk_trimming(monkeypatch):
    from tests.test_unit_gemma4_staged_prefill_integration import _forwarding_runner
    runner,config,captured,buffer,rope = _forwarding_runner(monkeypatch)
    try:
        runner._image_spans = ((1,4),)
        runner._launch_block(
            [1,2,3,4],tables={rope:(buffer(1),buffer(2))},
            masks={64:buffer(3),None:buffer(4)},kv_write_offset=0,
            key_begin_at=None,needs_logits=False,decode_phase=False,
        )
        assert not captured[0]['attention_mask_is_causal']
        assert captured[0]['window'] == 0
        assert captured[0]['prefill_attention_variants'] == ('gemma4_plain',)
        assert captured[1]['attention_mask_is_causal']
        assert captured[1]['window'] == 4
    finally:
        runner.close()


def test_public_multimodal_mtp_request_is_not_silently_downgraded():
    from hipengine import LLM, SamplingParams
    llm = object.__new__(LLM)
    llm._get_text_generator = lambda: SimpleNamespace(generate_multimodal_detailed=lambda *args: pytest.fail('downgraded'))
    with pytest.raises(NotImplementedError,match='multimodal speculative MTP'):
        llm.generate_multimodal_detailed('color?',np.zeros((8,8,3),np.uint8),SamplingParams(speculative_mtp=True))
