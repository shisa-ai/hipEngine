"""Minimal rocBLAS sgemm_batched correctness probe (gfx1151 bring-up)."""

import ctypes

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.rocblas import get_rocblas


def main() -> int:
    rt = get_hip_runtime()
    rb = get_rocblas()
    rb.set_workspace(0, 0)

    rng = np.random.default_rng(0)
    batch, m, n, k = 2, 3, 4, 5
    a_rm = rng.standard_normal((batch, m, k)).astype(np.float32)
    b_rm = rng.standard_normal((batch, k, n)).astype(np.float32)
    ref = np.stack([a_rm[i] @ b_rm[i] for i in range(batch)])  # (batch, m, n)

    # col-major storage: A_cm (m, k) ld=m ; B_cm (k, n) ld=k ; C_cm (m, n) ld=m
    a_cm = np.ascontiguousarray(np.stack([a_rm[i].T for i in range(batch)]))  # (batch, k, m) mem = (m,k) cm
    b_cm = np.ascontiguousarray(np.stack([b_rm[i].T for i in range(batch)]))  # (batch, n, k)
    abuf = malloc(a_cm.nbytes + 512)
    bbuf = malloc(b_cm.nbytes + 512)
    cbuf = malloc(ref.nbytes + 512)
    copy_host_to_device(abuf, host_array_ptr(a_cm), a_cm.nbytes)
    copy_host_to_device(bbuf, host_array_ptr(b_cm), b_cm.nbytes)
    rt.device_synchronize()

    ptrs = np.array([abuf.ptr + i * m * k * 4 for i in range(batch)], dtype=np.uint64)
    bptrs = np.array([bbuf.ptr + i * k * n * 4 for i in range(batch)], dtype=np.uint64)
    cptrs = np.array([cbuf.ptr + i * ldc * 4 for i in range(batch)], dtype=np.uint64) if False else \
        np.array([cbuf.ptr + i * m * n * 4 for i in range(batch)], dtype=np.uint64)
    pab = malloc(ptrs.nbytes + 512)
    pbb = malloc(bptrs.nbytes + 512)
    pcb = malloc(cptrs.nbytes + 512)
    copy_host_to_device(pab, host_array_ptr(ptrs), ptrs.nbytes)
    copy_host_to_device(pbb, host_array_ptr(bptrs), bptrs.nbytes)
    copy_host_to_device(pcb, host_array_ptr(cptrs), cptrs.nbytes)
    rt.device_synchronize()

    # NN: C_cm(m, n) = A_cm(m, k) @ B_cm(k, n)
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=batch, m=m, n=n, k=k, lda=m, ldb=k, ldc=m,
                     trans_a=False, trans_b=False)
    rt.device_synchronize()
    out = np.empty(batch * m * n, dtype=np.float32)
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": cbuf.ptr, "nbytes": out.nbytes})())
    got_cm = out.reshape(batch, n, m)  # memory = col-major (m, n)
    got = np.stack([got_cm[i].T for i in range(batch)])
    print("NN max|d|:", np.abs(got - ref).max())

    # NT: C_cm(m, n) = A_cm(m, k) @ B_cm(n, k)^T  (B stored as (n, k) col-major)
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=batch, m=m, n=n, k=k, lda=m, ldb=n, ldc=m,
                     trans_a=False, trans_b=True)
    rt.device_synchronize()
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": cbuf.ptr, "nbytes": out.nbytes})())
    got_cm = out.reshape(batch, n, m)
    got = np.stack([got_cm[i].T for i in range(batch)])
    print("NT max|d|:", np.abs(got - ref).max())

    # TN: C_cm(m, n) = A_cm(k, m)^T @ B_cm(k, n)  (A stored as (k, m) col-major)
    a_cm_t = np.ascontiguousarray(np.stack([a_rm[i] for i in range(batch)]))  # (batch, m, k) mem = (k, m) cm
    copy_host_to_device(abuf, host_array_ptr(a_cm_t), a_cm_t.nbytes)
    rt.device_synchronize()
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=batch, m=m, n=n, k=k, lda=k, ldb=k, ldc=m,
                     trans_a=True, trans_b=False)
    rt.device_synchronize()
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": cbuf.ptr, "nbytes": out.nbytes})())
    got_cm = out.reshape(batch, n, m)
    got = np.stack([got_cm[i].T for i in range(batch)])
    print("TN max|d|:", np.abs(got - ref).max())

    for buf in (abuf, bbuf, cbuf, pab, pbb, pcb):
        free(buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
