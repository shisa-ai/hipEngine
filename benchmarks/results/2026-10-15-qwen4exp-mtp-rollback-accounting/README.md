# Qwen4Exp MTP rollback accounting refresh

The current serial exact-target provider was exercised at proposal budget 4 across all rejection depths and full acceptance.

- Target forwards stop at the first mismatch: 1/2/3/4/4 for all-reject/reject-after-1/2/3/full-accept.
- Draft executes the full proposal before cursor trim.
- Rollback is cursor-only: dirty-destination restore bytes and restore allocations are zero.
- Per-cycle allocations are zero; one shared resident weight attachment is counted once.
- No snapshot/restore replay forwards or shared-weight double count were observed.

This refreshes MT-8 transaction evidence. It does not close MT-6: a real compact recurrent-state snapshot/restore resource cost is not charged by this serial cursor-trim provider.