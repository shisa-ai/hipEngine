import json
from types import SimpleNamespace
import numpy as np
import pytest


def tokenizer():
    from tokenizers import Tokenizer,models,decoders,pre_tokenizers,AddedToken
    from hipengine.tokenization.gguf import bytes_to_unicode
    from llguidance import LLTokenizer
    vocab = {value:key for key,value in bytes_to_unicode().items()}
    special = ["</s>","<tool_call>","</tool_call>","<think>","</think>"]
    vocab.update({token:256+i for i,token in enumerate(special)})
    encoder = Tokenizer(models.BPE(vocab=vocab,merges=[]))
    encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    encoder.decoder = decoders.ByteLevel()
    encoder.add_special_tokens([AddedToken(token,special=True) for token in special])
    return LLTokenizer(encoder.to_str(),n_vocab=261,eos_token=256)


def test_mask_precedes_argmax_and_enforces_enum():
    from hipengine.generation.grammar import GrammarSession
    grammar = GrammarSession(tokenizer(),{"format":"json_schema","value":{
        "type":"object","properties":{"x":{"const":7}},"required":["x"],"additionalProperties":False}})
    output = []
    desired = b'{"x":7}'
    for byte in desired:
        logits = np.zeros(261,np.float32)
        logits[ord("!")] = 100
        logits[byte] = 10
        chosen = grammar.select(logits)
        assert chosen == byte
        output.append(chosen)
    logits = np.zeros(261,np.float32)
    logits[256]=100
    assert grammar.select(logits)==256 and grammar.done
    assert json.loads(bytes(output))=={"x":7}


def test_xml_tool_schema_and_independent_matchers():
    from hipengine.generation.grammar import GrammarSession
    spec = {"format":"qwen4exp_tools","mode":"required","tools":[{"type":"function","function":{
        "name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string","enum":["A","B"]}},
                                    "required":["key"],"additionalProperties":False}}}]}
    first,second = GrammarSession(tokenizer(),spec),GrammarSession(tokenizer(),spec)
    text = b"<tool_call>\n<function=lookup>\n<parameter=key>\nA\n</parameter>\n</function>\n</tool_call>"
    for token in tokenizer().tokenize_bytes(text,parse_special=True):
        logits=np.zeros(261,np.float32); logits[token]=10
        assert first.select(logits)==token
    logits=np.zeros(261,np.float32); logits[256]=10
    assert first.select(logits)==256
    assert not second.done


def test_server_schema_does_not_use_json_tool_prefix_for_xml_engine():
    from hipengine.server.api import ChatCompletionRequest,_grammar_sampling_spec,_required_tool_sampling_forced_prefix
    engine=SimpleNamespace(supports_grammar=True,model_owned_tool_grammar=True)
    req=ChatCompletionRequest(model="test",messages=[{"role":"user","content":"x"}],
                              response_format={"type":"json_schema","json_schema":{"name":"x","schema":{
                                  "type":"object","properties":{"x":{"type":"integer"}}}}})
    assert _grammar_sampling_spec(req,engine)["format"]=="json_schema"
    assert _grammar_sampling_spec(req,None) is None
    assert _required_tool_sampling_forced_prefix(req,engine)==((),False)


def test_auto_tool_grammar_and_open_reasoning_special_tokens():
    from hipengine.generation.grammar import GrammarSession
    spec = {"format":"qwen4exp_tools","mode":"auto","tools":[{"type":"function","function":{
        "name":"ping","parameters":{"type":"object","properties":{}}}}]}
    for reasoning in (False,True):
        grammar = GrammarSession(tokenizer(),{**spec,"reasoning_open":reasoning})
        text = ("checking</think>" if reasoning else "")+"<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        tokens = tokenizer().tokenize_str(text,parse_special=True)
        for token in tokens:
            logits = np.zeros(261,np.float32); logits[token] = 10
            assert grammar.select(logits)==token
        logits = np.zeros(261,np.float32); logits[256] = 10
        assert grammar.select(logits)==256


@pytest.mark.parametrize("last_token,expected_reason,visible", [
    (ord("}"),"stop",(ord("{"),ord("}"))),
    (256,"eos",(ord("{"),)),
])
def test_grammar_completion_only_strips_actual_eos(last_token,expected_reason,visible):
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner
    owner = Qwen4ExpResidentServingRunner.__new__(Qwen4ExpResidentServingRunner)
    owner.generator = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=256))
    row = SimpleNamespace(
        generated_ids=[ord("{"),last_token],request=SimpleNamespace(),
        grammar=SimpleNamespace(done=True))
    finish = owner._finish(row)
    assert finish.reason==expected_reason
    assert owner._visible_ids(row,finish)==visible


def test_tools_and_schema_allow_only_call_or_valid_answer():
    from hipengine.generation.grammar import GrammarSession
    spec = {"format":"qwen4exp_tools","mode":"auto","tools":[{"type":"function","function":{
        "name":"ping","parameters":{"type":"object","properties":{}}}}],
        "answer":{"format":"json_schema","value":{"const":{"result":7}}}}
    for text in ('{"result":7}',"<tool_call>\n<function=ping>\n</function>\n</tool_call>"):
        grammar = GrammarSession(tokenizer(),spec)
        for token in tokenizer().tokenize_str(text,parse_special=True):
            scores=np.zeros(261,np.float32); scores[token]=10
            assert grammar.select(scores)==token
        scores=np.zeros(261,np.float32); scores[256]=10
        assert grammar.select(scores)==256


def test_xml_string_constraints_do_not_weaken_enums():
    from hipengine.generation.grammar import _string_rule
    assert _string_rule({"type":"string","enum":["a","abc"],"minLength":2})=='("abc")'
    with pytest.raises(ValueError):
        _string_rule({"type":"string","enum":["a"],"minLength":2})
    with pytest.raises(ValueError):
        _string_rule({"type":"string","not":{"const":"x"}})
    assert _string_rule({"type":"string","pattern":"foo"}).startswith("/(.|\\n)*")
    assert "\\$" in _string_rule({"type":"string","pattern":r"\$"})


def test_controlled_runner_feeds_selected_token_not_raw_argmax(monkeypatch):
    from hipengine.generation.engine_loop import EngineLoopConfig,SubmitPollTextGenerator
    from hipengine.generation.registry import GenerationRequest
    from tests.test_qwen4_exp_resident_serving import _generator
    generator = _generator()
    calls = []
    class Matcher:
        done = False
        def select(self,logits,**kwargs):
            token = 99 if calls else 42
            self.done = token == 99
            return token
    monkeypatch.setattr(generator,"_grammar_session",lambda request: Matcher())
    def prefill(tokens,**kwargs):
        assert kwargs["capture_logits"] is True
        return SimpleNamespace(token_id=7,logits=np.zeros(100,np.float32))
    def step(token,**kwargs):
        calls.append((token,kwargs))
        return SimpleNamespace(token_id=8,logits=np.zeros(100,np.float32))
    generator.runner.prefill = prefill
    generator.runner.step = step
    driver = SubmitPollTextGenerator(generator,config=EngineLoopConfig(max_active_requests=1))
    try:
        output = driver.generate_detailed(GenerationRequest(
            prompts=("x",),max_tokens=2,temperature=0.,top_p=1.,ignore_eos=False,
            grammar={"format":"choice","value":["x"]}))
        assert calls[0][0] == 42
        assert calls[0][1]["token_id_resident"] is False
        assert calls[0][1]["capture_logits"] is True
        assert output[0].text == "42"
    finally:
        driver.close()


def test_speculative_grammar_cannot_silently_bypass_mask():
    from hipengine.generation.engine_loop import SubmitPollTextGenerator
    from hipengine.generation.registry import GenerationRequest
    driver = SubmitPollTextGenerator.__new__(SubmitPollTextGenerator)
    request = GenerationRequest(prompts=("x",),max_tokens=2,temperature=0.,top_p=1.,
        ignore_eos=False,grammar={"format":"choice","value":["x"]})
    with pytest.raises(NotImplementedError,match="grammar"):
        driver.submit_speculative_detailed(request)
    with pytest.raises(NotImplementedError,match="grammar"):
        driver.submit_speculative_many_detailed((request,))


def test_single_tool_contract_masks_second_call():
    from hipengine.generation.grammar import GrammarSession
    spec={"format":"qwen4exp_tools","mode":"auto","parallel":False,"tools":[{"type":"function","function":{
        "name":"ping","parameters":{"type":"object","properties":{}}}}]}
    grammar=GrammarSession(tokenizer(),spec)
    for token in tokenizer().tokenize_str(
        "Checking\n<tool_call>\n<function=ping>\n</function>\n</tool_call>",parse_special=True):
        scores=np.zeros(261,np.float32); scores[token]=10
        assert grammar.select(scores)==token
    mask=np.frombuffer(grammar.matcher.compute_logit_bias(),dtype=np.uint8)
    assert mask[257]==0 and mask[256]>0


def test_direct_unsupported_modes_reject_grammar():
    from hipengine.generation.registry import GenerationRequest
    from tests.test_qwen4_exp_resident_serving import _generator
    generator = _generator()
    request = GenerationRequest(prompts=("x",),max_tokens=2,temperature=0.,top_p=1.,
        ignore_eos=False,grammar={"format":"choice","value":["x"]})
    try:
        with pytest.raises(NotImplementedError,match="grammar"):
            generator.generate_speculative_detailed(request)
        with pytest.raises(NotImplementedError,match="grammar"):
            list(generator.stream_speculative_detailed(request))
        with pytest.raises(NotImplementedError,match="grammar"):
            generator.generate_multimodal_detailed("x",None,request)
    finally:
        generator.close()
