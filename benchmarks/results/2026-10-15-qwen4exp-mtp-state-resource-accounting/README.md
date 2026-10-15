# Qwen4Exp MTP state-row verifier resource accounting

This packet accounts the workspace allocated by `Qwen4ExpGGUFResidentModelRunner._verify_rows_workspace` for the rows 2-8 state-row verifier. It uses the current Qwen3.8-Flash-Next UD-Q4_K_XL GGUF geometry and counts GDN recurrent matrix rows, convolution state rows, PLE output rows, and argmax IDs.

| Candidate budget | Verifier rows | State-row workspace |
|---:|---:|---:|
| 1 | 2 | 227.33 MiB |
| 2 | 3 | 340.99 MiB |
| 3 | 4 | 454.66 MiB |
| 5 | 6 | 681.98 MiB |
| 7 | 8 | 909.31 MiB |

Budget 3 is the current long-context economics configuration and therefore carries a 476,741,664-byte workspace charge. This is a real verifier-state resource charge, distinct from the current provider's draft cursor-trim rollback, which has zero restore bytes. The provider does not currently snapshot and restore these target states; no production admission change is made.

Reproducer: `scripts/qwen4exp_mtp_state_resource_accounting.py`.
