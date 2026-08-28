"""Basic one-image Qwen3-VL vision encoder for Qwen4Exp."""

from __future__ import annotations

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.vision.qwen4_exp_vision import qwen4_exp_vision_add_bias_residual_f32,qwen4_exp_vision_attention_f32,qwen4_exp_vision_bias_gelu_tanh_f32,qwen4_exp_vision_layernorm_f32
from hipengine.loading.qwen4_exp_vision_materialize import Qwen4ExpVisionResidentWeights
from hipengine.runtime.gguf_linear import GGUF_ACTIVATION_F32,GGUF_OUTPUT_F32,launch_gguf_linear


class Qwen4ExpVisionRunner:
    """Encode exactly one 32x32 RGB image to one H2560 image-token row."""
    def __init__(self,resident:Qwen4ExpVisionResidentWeights,*,patch_weight0:np.ndarray,patch_weight1:np.ndarray,patch_bias:np.ndarray,position_embedding:np.ndarray):
        self.resident=resident;self.config=resident.plan.config;self.runtime=resident.runtime;self.closed=False
        self.patch_weight0=np.ascontiguousarray(patch_weight0,dtype=np.float32);self.patch_weight1=np.ascontiguousarray(patch_weight1,dtype=np.float32);self.patch_bias=np.ascontiguousarray(patch_bias,dtype=np.float32);self.position_embedding=np.ascontiguousarray(position_embedding,dtype=np.float32)
        if self.patch_weight0.shape!=(1152,3,16,16) or self.patch_weight1.shape!=self.patch_weight0.shape or self.patch_bias.shape!=(1152,) or self.position_embedding.shape!=(2304,1152):raise ValueError('invalid Qwen4Exp vision patch/position tensors')
        self._buffers=[]
        for elements in (4*1152,4*1152,4*3456,4*1152,4*1152,4*4304,4*4304,4*1152,4,4,4608,4608,2560):self._buffers.append(malloc(elements*4,runtime=self.runtime))
        ph=np.array([0,0,1,1],np.int32);pw=np.array([0,1,0,1],np.int32)
        copy_host_to_device(self._buffers[8],host_array_ptr(ph),runtime=self.runtime);copy_host_to_device(self._buffers[9],host_array_ptr(pw),runtime=self.runtime)
    def _w(self,name):return self.resident.weight(name)
    def _p(self,name):return self._w(name).allocation('raw').tensor.ptr
    def preprocess(self,image)->np.ndarray:
        x=np.asarray(image)
        if x.shape!=(32,32,3):raise ValueError('basic Qwen4Exp vision currently requires one 32x32 RGB image')
        if x.dtype==np.uint8:x=x.astype(np.float32)/127.5-1.0
        else:
            x=x.astype(np.float32)
            if x.min()>=0 and x.max()<=1:x=x*2-1
        patches=[]
        for y,x0 in ((0,0),(0,16),(16,0),(16,16)):
            patch=np.transpose(x[y:y+16,x0:x0+16],(2,0,1))
            value=np.einsum('ochw,chw->o',self.patch_weight0,patch,optimize=True)+np.einsum('ochw,chw->o',self.patch_weight1,patch,optimize=True)+self.patch_bias
            patches.append(value)
        out=np.stack(patches).astype(np.float32)
        # align-corners 48x48 -> 2x2 selects the four learned-position corners.
        out+=self.position_embedding[[0,47,48*47,48*48-1]]
        return np.ascontiguousarray(out)
    def encode(self,image)->np.ndarray:
        if self.closed:raise RuntimeError('Qwen4Exp vision runner is closed')
        rows=4;h=1152;inter=4304
        initial=self.preprocess(image);copy_host_to_device(self._buffers[0],host_array_ptr(initial),runtime=self.runtime);cur=self._buffers[0];norm=self._buffers[1];qkv=self._buffers[2];attn=self._buffers[3];projected=self._buffers[4];ff1=self._buffers[5];gelu=self._buffers[6];ff2=self._buffers[7]
        for layer in range(27):
            p=f'layers.{layer}.'
            qwen4_exp_vision_layernorm_f32(cur.ptr,self._p(p+'ln1.weight'),self._p(p+'ln1.bias'),norm.ptr,rows,h,self.config.norm_epsilon,runtime=self.runtime)
            launch_gguf_linear(self._w(p+'attn_qkv.weight'),norm.ptr,qkv.ptr,rows,h,3*h,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
            qwen4_exp_vision_attention_f32(qkv.ptr,self._p(p+'attn_qkv.bias'),self._buffers[8].ptr,self._buffers[9].ptr,attn.ptr,rows,runtime=self.runtime)
            launch_gguf_linear(self._w(p+'attn_out.weight'),attn.ptr,projected.ptr,rows,h,h,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
            qwen4_exp_vision_add_bias_residual_f32(projected.ptr,self._p(p+'attn_out.bias'),cur.ptr,projected.ptr,rows,h,runtime=self.runtime)
            qwen4_exp_vision_layernorm_f32(projected.ptr,self._p(p+'ln2.weight'),self._p(p+'ln2.bias'),norm.ptr,rows,h,self.config.norm_epsilon,runtime=self.runtime)
            launch_gguf_linear(self._w(p+'ffn_up.weight'),norm.ptr,ff1.ptr,rows,h,inter,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
            qwen4_exp_vision_bias_gelu_tanh_f32(ff1.ptr,self._p(p+'ffn_up.bias'),gelu.ptr,rows,inter,runtime=self.runtime)
            launch_gguf_linear(self._w(p+'ffn_down.weight'),gelu.ptr,ff2.ptr,rows,inter,h,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
            qwen4_exp_vision_add_bias_residual_f32(ff2.ptr,self._p(p+'ffn_down.bias'),projected.ptr,cur.ptr,rows,h,runtime=self.runtime)
        qwen4_exp_vision_layernorm_f32(cur.ptr,self._p('post_norm.weight'),self._p('post_norm.bias'),norm.ptr,rows,h,self.config.norm_epsilon,runtime=self.runtime)
        # Four merge-tile-ordered patch rows are already one contiguous 4608 row.
        self.runtime.memcpy(self._buffers[10].ptr,norm.ptr,4608*4,HipMemcpyKind.DEVICE_TO_DEVICE)
        launch_gguf_linear(self._w('merge.fc1.weight'),self._buffers[10].ptr,self._buffers[11].ptr,1,4608,4608,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
        qwen4_exp_vision_bias_gelu_tanh_f32(self._buffers[11].ptr,self._p('merge.fc1.bias'),self._buffers[10].ptr,1,4608,runtime=self.runtime)
        launch_gguf_linear(self._w('merge.fc2.weight'),self._buffers[10].ptr,self._buffers[12].ptr,1,4608,2560,activation_dtype=GGUF_ACTIVATION_F32,output_dtype=GGUF_OUTPUT_F32,runtime=self.runtime)
        qwen4_exp_vision_add_bias_residual_f32(self._buffers[12].ptr,self._p('merge.fc2.bias'),0,self._buffers[12].ptr,1,2560,runtime=self.runtime)
        self.runtime.device_synchronize();out=np.empty((1,2560),np.float32);copy_device_to_host(host_array_ptr(out),self._buffers[12],runtime=self.runtime);return out
    def close(self):
        if self.closed:return
        for b in reversed(self._buffers):free(b,runtime=self.runtime)
        self._buffers.clear();self.closed=True

__all__=['Qwen4ExpVisionRunner']
