"""Gemma image chat policy and sampling through the shared HTTP route."""
import base64
import io

import numpy as np
import pytest

pytest.importorskip('fastapi')
from PIL import Image

from hipengine.generation import FinishDetails, GenerationOutput
from hipengine.generation.registry import TokenLogprob
from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator
from tests.test_integration_surya_server_multimodal import _engine, _client


class ImageGenerator:
    supports_vision = True
    vision_prompt_marker = Gemma4GGUFGenerator.vision_prompt_marker
    vision_media_input = Gemma4GGUFGenerator.vision_media_input
    vision_uses_chat_template = Gemma4GGUFGenerator.vision_uses_chat_template
    chat_template_family = 'gemma'
    max_sequence_length = 1024

    def __init__(self):
        self.calls = []

    def render_chat_prompt(self,messages,*,enable_thinking=True,**kwargs):
        return '<|turn>user\n'+messages[0].content+'<turn|>\n<|turn>model\n'+str(enable_thinking)

    def count_tokens(self,text):
        return len(str(text).split())

    def count_multimodal_tokens(self,prompt,media):
        return 291

    def generate_multimodal_detailed(self,prompt,media,request):
        self.calls.append((prompt,media,request))
        return GenerationOutput(text='red',generated_token_ids=(7,),prompt_tokens=291,
                                token_logprobs=(TokenLogprob(token_id=7,token_text='red',logprob=-0.2),),
                                finish_details=FinishDetails(reason='length',sampler_mode='greedy'))

    def close(self):
        pass


def test_http_image_chat_preserves_thinking_sampling_and_image_order():
    image = np.full((4,6,3),(255,0,0),dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf,format='PNG')
    url = 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    generator = ImageGenerator()
    with _client(_engine(generator),model='gemma4') as client:
        response = client.post('/v1/chat/completions',json={
            'model':'gemma4','messages':[{'role':'user','content':[
                {'type':'text','text':'Which color? '},
                {'type':'image_url','image_url':{'url':url}},
            ]}],'max_tokens':3,'temperature':0,'seed':77,'min_p':0.1,
            'repetition_penalty':1.1,'chat_template_kwargs':{'enable_thinking':False},
            'logprobs':True,
        })
    assert response.status_code == 200, response.text
    assert response.json()['usage']['prompt_tokens'] == 291
    assert response.json()['choices'][0]['logprobs']['content'][0]['token'] == 'red'
    prompt,media,request = generator.calls[0]
    assert '<|image|>' in prompt and '<|vision_start|>' not in prompt
    assert prompt.startswith('<|turn>') and prompt.endswith('False')
    np.testing.assert_array_equal(media['items'][0]['data'],image)
    assert request.seed == 77 and request.min_p == 0.1 and request.repetition_penalty == 1.1
    assert request.max_tokens == 3


@pytest.mark.parametrize('default_cap',[None,4096,100,0])
def test_http_image_auto_budget_uses_expanded_prompt(default_cap):
    buf = io.BytesIO()
    Image.new('RGB',(4,6),'red').save(buf,format='PNG')
    url = 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    generator = ImageGenerator()
    with _client(_engine(generator),model='gemma4',chat_default_max_tokens=default_cap,
                 max_context_tokens=1024) as client:
        response = client.post('/v1/chat/completions',json={
            'model':'gemma4','messages':[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':url}},
                {'type':'text','text':'Color?'},
            ]}],'temperature':0,'chat_template_kwargs':{'enable_thinking':False},
        })
    assert response.status_code == 200,response.text
    assert generator.calls[0][2].max_tokens == min(8192 if default_cap is None else default_cap,1024-291-1)
