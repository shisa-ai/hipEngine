"""Real Gemma 4 image generation; set HIPENGINE_GEMMA4_GGUF and _MMPROJ."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL('libamdhip64.so')
except OSError:
    pytest.skip('HIP runtime is unavailable', allow_module_level=True)


@pytest.fixture(scope='module')
def artifacts():
    paths = tuple(Path(os.environ.get(name,'')) for name in (
        'HIPENGINE_GEMMA4_GGUF','HIPENGINE_GEMMA4_MMPROJ',
    ))
    if not all(path.is_file() for path in paths):
        pytest.skip('set HIPENGINE_GEMMA4_GGUF and HIPENGINE_GEMMA4_MMPROJ to real artifacts')
    return paths


def test_real_sidecar_matches_cpu(artifacts):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.gemma4_vision_gguf import build_gemma4_vision_gguf_map, read_gemma4_vision_weights
    from hipengine.loading.gemma4_image import preprocess_gemma4_image
    from hipengine.kernels.cpu_reference.gemma4_vision import (
        gemma4_vision_geometry_from_config, gemma4_vision_weights_from_slots,
        gemma4_vision_forward_positions,
    )
    from hipengine.runtime.gemma4_vision import Gemma4VisionRunner

    reader = GGUFReader(artifacts[1])
    model_map = build_gemma4_vision_gguf_map(reader.info)
    c = model_map.config
    geometry = gemma4_vision_geometry_from_config(c)
    weights = gemma4_vision_weights_from_slots(read_gemma4_vision_weights(reader.info,model_map),geometry)
    runner = Gemma4VisionRunner(reader,c,backend='hip_gfx1100')
    try:
        for shape,budget in (((48,48),1),((137,211),5),((96,144),70),((144,96),280)):
            pixels = np.random.default_rng(40).integers(0,256,(*shape,3),dtype=np.uint8)
            image = preprocess_gemma4_image(pixels,patch_size=c.patch_size,max_soft_tokens=budget)
            pos = np.stack((image.pos_x,image.pos_y),axis=-1)
            cpu = gemma4_vision_forward_positions(weights,image.patches_chw,pos,geometry,grid_size=image.grid_size).embeddings
            gpu = runner.forward(image.patches_chw,pos,image.grid_size)
            np.testing.assert_allclose(gpu,cpu,atol=0.002,rtol=0.002)
    finally:
        runner.close()


def test_public_images_recognize_color_and_restore_text_path(artifacts):
    from hipengine import LLM, SamplingParams

    llm = LLM(model=str(artifacts[0]),vision_model=str(artifacts[1]),backend='hip_gfx1100',
              max_sequence_length=1024,speculative_mtp_serving='off')
    params = SamplingParams(max_tokens=16,temperature=0)
    try:
        for name,rgb in (('red',(255,0,0)),('blue',(0,0,255))):
            pixels = np.full((96,144,3),rgb,dtype=np.uint8)
            prompt = ('<|turn>user\n<|image|>\nWhat is the dominant color in this image? '
                      'Reply with just the color name.<turn|>\n<|turn>model\n'
                      '<|channel>thought\n<channel|>')
            result = llm.generate_multimodal_detailed(prompt,pixels,params)
            assert llm.supports_vision
            assert name in result.text.lower(), result.text
            assert result.prompt_tokens > 200
            assert result.finish_details.sampler_mode == 'greedy'
        media = {'items':[
            {'type':'image','data':np.full((96,144,3),(255,0,0),dtype=np.uint8)},
            {'type':'image','data':np.full((144,96,3),(0,0,255),dtype=np.uint8)},
        ]}
        prompt = ('<|turn>user\nFirst: <|image|>\nSecond: <|image|>\n'
                  'Name the dominant color in each image, in order. Reply with just two color names.'
                  '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')
        result = llm.generate_multimodal_detailed(prompt,media,params)
        assert result.prompt_tokens > 512
        text = result.text.lower()
        assert 'red' in text and 'blue' in text and text.index('red') < text.index('blue'), result.text
        prompt = ('<|turn>user\nWhat is 2 + 2? Reply with just the number.'
                  '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')
        assert '4' in llm.generate([prompt],params)[0]
    finally:
        llm.close()
