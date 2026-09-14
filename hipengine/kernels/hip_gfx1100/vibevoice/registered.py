"""Shared gfx11 VibeVoice registrations, installed by each backend peer."""
from hipengine.kernels.registry import KernelKey, register, is_registered
from hipengine.kernels.hip_gfx1100.vibevoice import encoder
from hipengine.kernels.hip_gfx1100.convert import cast
from hipengine.kernels.hip_gfx1100.fused import paro_silu
from hipengine.kernels.hip_gfx1100.linear import dense_gemv
from hipengine.kernels.hip_gfx1100.rotary import qwen35_rotary


def frontend_gemm(x,w,out,rows,inputs,outputs,*,runtime):
    fn = dense_gemv.dense_prefill_wmma_out_bf16 if outputs % 128 == 0 and inputs % 32 == 0 else dense_gemv.dense_prefill_gemm_out_bf16
    fn(x,w,out,rows,inputs,outputs,stream=0,runtime=runtime)


def frontend_gemm_strict(x,w,out,rows,inputs,outputs,*,runtime):
    dense_gemv.dense_prefill_gemm_out_bf16(x,w,out,rows,inputs,outputs,stream=0,runtime=runtime)


def incremental_prefill(runner,hidden_rows,rows,start):
    from hipengine.core.runtime import MemcpyKind
    width = runner.spec.hidden_size*2
    for row in range(rows):
        runner.runtime.memcpy(runner._hidden.ptr,hidden_rows.ptr+row*width,width,MemcpyKind.DEVICE_TO_DEVICE)
        runner.forward_layers(start+row)
        runner.runtime.memcpy(hidden_rows.ptr+row*width,runner._hidden.ptr,width,MemcpyKind.DEVICE_TO_DEVICE)


def batched_prefill(runner,hidden_rows,rows,start):
    runner._prefill_batched(hidden_rows,rows,start)


def register_vibevoice_kernels(backend):
    from hipengine.kernels.vibevoice import PRIMITIVES
    modules=(encoder,cast,paro_silu,dense_gemv,qwen35_rotary)
    for name in PRIMITIVES:
        fn=next(getattr(module,name) for module in modules if hasattr(module,name))
        key=KernelKey(backend,name,'bf16','strict')
        if not is_registered(key):register(key,fn)
    for layer,variant,fn in (
        ('vibevoice_frontend_gemm','wmma',frontend_gemm),
        ('vibevoice_frontend_gemm','strict',frontend_gemm_strict),
        ('vibevoice_prefill','strict',incremental_prefill),
        ('vibevoice_prefill','hipblaslt',batched_prefill),
    ):
        key=KernelKey(backend,layer,'bf16',variant)
        if not is_registered(key):register(key,fn)
