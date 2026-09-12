from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc


def _hip_available():
    try: ctypes.CDLL('libamdhip64.so')
    except OSError: return False
    return True


def _vision_rope(row, h, w):
    out=row.copy()
    scale=10000.0 ** (-2.0/36.0)
    for p in range(36):
        local=p if p<18 else p-18; pos=h if p<18 else w; theta=pos*(scale**local);c=np.cos(theta);s=np.sin(theta);x0=row[p];x1=row[p+36];out[p]=x0*c-x1*s;out[p+36]=x0*s+x1*c
    return out


@pytest.mark.skipif(not _hip_available(), reason='HIP runtime unavailable')
def test_qwen4_exp_vision_helpers_match_numpy():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.vision.qwen4_exp_vision import qwen4_exp_vision_layernorm_f32,qwen4_exp_vision_add_bias_residual_f32,qwen4_exp_vision_bias_gelu_tanh_f32,qwen4_exp_vision_attention_f32
    rt=get_hip_runtime();rng=np.random.default_rng(38);tokens=4;features=1152
    x=rng.normal(size=(tokens,features)).astype(np.float32);w=rng.normal(size=features).astype(np.float32);b=rng.normal(size=features).astype(np.float32);res=rng.normal(size=x.shape).astype(np.float32)
    qkv=rng.normal(0,.1,size=(tokens,3*features)).astype(np.float32);qb=rng.normal(0,.1,size=3*features).astype(np.float32);ph=np.array([0,0,1,1],np.int32);pw=np.array([0,1,0,1],np.int32)
    hosts=[x,w,b,res,qkv,qb,ph,pw];dev=[];outs=[]
    try:
        for a in hosts:
            d=malloc(a.nbytes,runtime=rt);copy_host_to_device(d,host_array_ptr(a),runtime=rt);dev.append(d)
        for shape in (x.shape,x.shape,x.shape,x.shape):outs.append(malloc(np.prod(shape)*4,runtime=rt))
        qwen4_exp_vision_layernorm_f32(dev[0].ptr,dev[1].ptr,dev[2].ptr,outs[0].ptr,tokens,features,runtime=rt)
        qwen4_exp_vision_add_bias_residual_f32(dev[0].ptr,dev[2].ptr,dev[3].ptr,outs[1].ptr,tokens,features,runtime=rt)
        qwen4_exp_vision_bias_gelu_tanh_f32(dev[0].ptr,dev[2].ptr,outs[2].ptr,tokens,features,runtime=rt)
        qwen4_exp_vision_attention_f32(dev[4].ptr,dev[5].ptr,dev[6].ptr,dev[7].ptr,outs[3].ptr,tokens,runtime=rt)
        got=[]
        for d in outs:
            a=np.empty(x.shape,np.float32);copy_device_to_host(host_array_ptr(a),d,runtime=rt);got.append(a)
    finally:
        for d in reversed(outs):free(d,runtime=rt)
        for d in reversed(dev):free(d,runtime=rt)
    mean=x.mean(1,keepdims=True,dtype=np.float32);var=((x-mean)**2).mean(1,keepdims=True,dtype=np.float32);ln=(x-mean)/np.sqrt(var+1e-6)*w+b
    np.testing.assert_allclose(got[0],ln,rtol=2e-5,atol=2e-5);np.testing.assert_allclose(got[1],x+b+res,rtol=1e-6,atol=1e-6)
    z=x+b;gelu=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)));np.testing.assert_allclose(got[2],gelu,rtol=2e-6,atol=2e-6)
    q=(qkv[:,:features]+qb[:features]).reshape(tokens,16,72);k=(qkv[:,features:2*features]+qb[features:2*features]).reshape(tokens,16,72);v=(qkv[:,2*features:]+qb[2*features:]).reshape(tokens,16,72);expected=np.empty_like(v)
    for qi in range(tokens):
      for h in range(16):
        qr=_vision_rope(q[qi,h],ph[qi],pw[qi]);kr=np.stack([_vision_rope(k[j,h],ph[j],pw[j]) for j in range(tokens)]);scores=kr@qr/np.sqrt(72);prob=np.exp(scores-scores.max());prob/=prob.sum();expected[qi,h]=prob@v[:,h]
    np.testing.assert_allclose(got[3],expected.reshape(tokens,features),rtol=2e-5,atol=2e-5)
