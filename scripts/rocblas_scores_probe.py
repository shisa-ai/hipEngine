"""Replicate the Surya scores-GEMM geometry exactly (stacked C tiles, strided B)."""

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.rocblas import get_rocblas


def main() -> int:
    rt = get_hip_runtime()
    rb = get_rocblas()
    rb.set_workspace(0, 0)

    rng = np.random.default_rng(1)
    nq, nk, hd, tokens = 8, 2, 16, 24
    repeat = nq // nk
    plane = tokens * hd
    q_row = nq * hd

    k_planes = rng.standard_normal((nk, tokens, hd)).astype(np.float32)   # row-major planes
    q_rm = rng.standard_normal((tokens, nq * hd)).astype(np.float32)      # row-major
    scores = np.zeros((nq, tokens, tokens), dtype=np.float32)

    kbuf = malloc(k_planes.nbytes + 512)
    qbuf = malloc(q_rm.nbytes + 512)
    sbuf = malloc(scores.nbytes + 512)
    copy_host_to_device(kbuf, host_array_ptr(k_planes), k_planes.nbytes)
    copy_host_to_device(qbuf, host_array_ptr(q_rm), q_rm.nbytes)
    rt.device_synchronize()

    aptrs = np.array([kbuf.ptr + (h // repeat) * plane * 4 for h in range(nq)], dtype=np.uint64)
    bptrs = np.array([qbuf.ptr + h * hd * 4 for h in range(nq)], dtype=np.uint64)
    cptrs = np.array([sbuf.ptr + h * tokens * 4 for h in range(nq)], dtype=np.uint64)
    pab = malloc(aptrs.nbytes + 512)
    pbb = malloc(bptrs.nbytes + 512)
    pcb = malloc(cptrs.nbytes + 512)
    copy_host_to_device(pab, host_array_ptr(aptrs), aptrs.nbytes)
    copy_host_to_device(pbb, host_array_ptr(bptrs), bptrs.nbytes)
    copy_host_to_device(pcb, host_array_ptr(cptrs), cptrs.nbytes)
    rt.device_synchronize()

    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=nq, m=tokens, n=tokens, k=hd,
                     lda=hd, ldb=q_row, ldc=tokens,
                     trans_a=True, trans_b=False)
    rt.device_synchronize()
    out = np.empty(nq * tokens * tokens, dtype=np.float32)
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": sbuf.ptr, "nbytes": out.nbytes})())
    got = out.reshape(nq, tokens, tokens)  # row-major view used by the mask/softmax kernels

    ref = np.stack([
        k_planes[h // repeat] @ q_rm[:, h * hd:(h + 1) * hd].T for h in range(nq)
    ])  # ref[h][i, j] = k_i . q_j
    print("as-is   max|d|:", np.abs(got - ref).max())
    # variant: unique A pointers (per-query-head k copies in separate storage)
    kbig = np.ascontiguousarray(np.repeat(k_planes, repeat, axis=0))  # (nq, tokens, hd)
    kbuf2 = malloc(kbig.nbytes + 512)
    copy_host_to_device(kbuf2, host_array_ptr(kbig), kbig.nbytes)
    aptrs2 = np.array([kbuf2.ptr + h * plane * 4 for h in range(nq)], dtype=np.uint64)
    copy_host_to_device(pab, host_array_ptr(aptrs2), aptrs2.nbytes)
    rt.device_synchronize()
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=nq, m=tokens, n=tokens, k=hd,
                     lda=hd, ldb=q_row, ldc=tokens,
                     trans_a=True, trans_b=False)
    rt.device_synchronize()
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": sbuf.ptr, "nbytes": out.nbytes})())
    got2 = out.reshape(nq, tokens, tokens)
    print("unique-A as-is max|d|:", np.abs(got2 - ref).max(),
          "transp:", np.abs(got2.transpose(0, 2, 1) - ref).max())
    free(kbuf2)
    print("transp  max|d|:", np.abs(got.transpose(0, 2, 1) - ref).max())

    # variant B: contiguous C tiles (offset h*tokens^2), strided B — same lda/ldb/ldc
    cptrs_b = np.array([sbuf.ptr + h * tokens * tokens * 4 for h in range(nq)], dtype=np.uint64)
    copy_host_to_device(pcb, host_array_ptr(cptrs_b), cptrs_b.nbytes)
    rt.device_synchronize()
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=nq, m=tokens, n=tokens, k=hd,
                     lda=hd, ldb=q_row, ldc=tokens,
                     trans_a=True, trans_b=False)
    rt.device_synchronize()
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": sbuf.ptr, "nbytes": out.nbytes})())
    got3 = out.reshape(nq, tokens, tokens)
    print("contigC as-is:", np.abs(got3 - ref).max(),
          "transp:", np.abs(got3.transpose(0, 2, 1) - ref).max())

    # variant C: contiguous B blocks, stacked C (original C layout)
    qbig = np.ascontiguousarray(np.stack([q_rm[:, h*hd:(h+1)*hd] for h in range(nq)]))
    qbuf2 = malloc(qbig.nbytes + 512)
    copy_host_to_device(qbuf2, host_array_ptr(qbig), qbig.nbytes)
    bptrs_c = np.array([qbuf2.ptr + h * tokens * hd * 4 for h in range(nq)], dtype=np.uint64)
    copy_host_to_device(pbb, host_array_ptr(bptrs_c), bptrs_c.nbytes)
    copy_host_to_device(pcb, host_array_ptr(cptrs), cptrs.nbytes)
    rt.device_synchronize()
    rb.sgemm_batched(pab.ptr, pbb.ptr, pcb.ptr,
                     batch=nq, m=tokens, n=tokens, k=hd,
                     lda=hd, ldb=hd, ldc=tokens,
                     trans_a=True, trans_b=False)
    rt.device_synchronize()
    copy_device_to_host(host_array_ptr(out),
                        type("V", (), {"ptr": sbuf.ptr, "nbytes": out.nbytes})())
    got4 = out.reshape(nq, tokens, tokens)
    print("contigB as-is:", np.abs(got4 - ref).max(),
          "transp:", np.abs(got4.transpose(0, 2, 1) - ref).max())
    free(qbuf2)
    for buf in (kbuf, qbuf, sbuf, pab, pbb, pcb):
        free(buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
