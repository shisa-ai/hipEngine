"""Recording boundaries and causal chunk carry, using real frontend weights."""
import numpy as np
import pytest
from tests._rocm_guard import hip_runtime_available

if not hip_runtime_available():
    pytest.skip('no usable HIP runtime',allow_module_level=True)

from hipengine.loading.vibevoice_asr import load_vibevoice_encoder,load_vibevoice_connector
from hipengine.runtime.vibevoice_encoder import VibevoiceFrontendRuntime


@pytest.fixture(scope='module')
def frontend():
    from hipengine.loading.hf_cache import resolve_model_path
    try:
        path = resolve_model_path('microsoft/VibeVoice-ASR-HF')
    except (FileNotFoundError,ValueError):
        pytest.skip('HF checkpoint unavailable')
    ac = load_vibevoice_encoder(path,'acoustic')
    se = load_vibevoice_encoder(path,'semantic')
    r = VibevoiceFrontendRuntime(*ac,*se,load_vibevoice_connector(path,'acoustic'),load_vibevoice_connector(path,'semantic'))
    yield r
    r.close()


def test_chunk_carry_and_recording_reset(frontend):
    pcm = np.random.default_rng(18).normal(0,.1,12800).astype(np.float32)
    whole = frontend.encode(pcm,chunk_samples=12800)
    chunked = frontend.encode(pcm,chunk_samples=3200)
    for key in whole:
        np.testing.assert_allclose(chunked[key],whole[key],atol=0,rtol=0)
    frontend.encode(pcm[::-1].copy(),chunk_samples=3200)
    repeated = frontend.encode(pcm,chunk_samples=3200)
    for key in whole:
        np.testing.assert_array_equal(repeated[key],chunked[key])


@pytest.mark.parametrize('samples',[3199,3200,3201])
def test_partial_final_frame(frontend,samples):
    pcm = np.zeros(samples,np.float32)
    result = frontend.encode(pcm,chunk_samples=3200)
    for key,latents in result.items():
        assert latents.shape == ((samples+3199)//3200,frontend.specs[key].hidden_size)
        assert np.isfinite(latents).all()


def test_sampling_after_join(frontend):
    rng = np.random.default_rng(19)
    pcm = rng.normal(0,.1,9600).astype(np.float32)
    noise = rng.normal(size=(3,64)).astype(np.float32)
    whole = frontend.forward(pcm,noise=noise,noise_scale=.25,chunk_samples=9600)
    chunks = frontend.forward(pcm,noise=noise,noise_scale=.25,chunk_samples=3200)
    np.testing.assert_array_equal(whole,chunks)


def test_sixty_second_transition_against_torch_gpu(frontend):
    from pathlib import Path
    path = Path(__file__).parent/'fixtures/vibevoice_asr_gpu/vibevoice_asr_chunk.npz'
    if not path.is_file():
        pytest.skip('torch chunk fixture unavailable')
    with np.load(path) as f:
        result = frontend.encode(f['pcm_full'],chunk_samples=1_440_000)
        for name,got in result.items():
            ref = f[name+'_latent_chunked'][0]
            assert got.shape == ref.shape
            assert np.isfinite(got).all()
            assert np.abs(got-ref).max()/max(np.abs(ref).max(),1e-9) <= .05
