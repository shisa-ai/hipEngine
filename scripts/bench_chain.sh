#!/usr/bin/env bash
# Run measurement arms sequentially with per-arm evidence, and fail loudly.
#
# Motivation: three A/B chains on 2026-08-30 printed an unconditional
# "<something>_DONE" marker while an arm had actually died (one with
# "benchmark requires tracked-clean source", two writing no artifact at all), and the
# surviving 14-byte logs made a failed chain look like a completed measurement.
#
# Contract per arm:
#   * stdout+stderr is tee'd to OUTDIR/<name>.log (never a 14-byte marker)
#   * the exit code is captured and reported
#   * if the arm wrote $CHAIN_JSON, the runner requires status == "complete"
#   * the chain exits non-zero if any arm failed, so a background task that says
#     CHAIN_DONE genuinely means every arm measured
#
# Usage:
#   scripts/bench_chain.sh TAG OUTDIR "NAME|COMMAND" ["NAME|COMMAND" ...]
#
# The command may reference $CHAIN_JSON for its artifact path (derived as
# OUTDIR/TAG-NAME.json), e.g.
#
#   scripts/bench_chain.sh floor /tmp/he-chain \
#     'floor33|python scripts/gguf_mtp_c1c8_server_bench.py --json "$CHAIN_JSON"' \
#     'floor16|python scripts/gguf_probe_floors.py 16 --json "$CHAIN_JSON"'
#
# An arm whose name ends in "!" is not required to write an artifact (for arms like
# a pytest run); every other arm must produce $CHAIN_JSON with status "complete".
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 TAG OUTDIR \"NAME|COMMAND\" [...]" >&2
  exit 2
fi

TAG="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR" || exit 2

names=()
rcs=()
verdicts=()

for spec in "$@"; do
  name="${spec%%|*}"
  cmd="${spec#*|}"
  if [[ "$name" == "$spec" || -z "$cmd" ]]; then
    echo "arm spec must be NAME|COMMAND: $spec" >&2
    exit 2
  fi

  require_json=1
  if [[ "${name: -1}" == "!" ]]; then
    require_json=0
    name="${name%!}"
  fi
  export CHAIN_JSON="$OUTDIR/$TAG-$name.json"
  log="$OUTDIR/$TAG-$name.log"
  : > "$log"

  echo "== arm $name start $(date -u +%H:%M:%S) json=$CHAIN_JSON"
  # shellcheck disable=SC2091
  ( eval "$cmd" ) 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}

  verdict="ok"
  if [[ $rc -ne 0 ]]; then
    verdict="exit=$rc"
  elif [[ -f "$CHAIN_JSON" ]]; then
    status=$(python3 - "$CHAIN_JSON" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("status", "no-status-field"))
except Exception as exc:  # unreadable artifact is a failure, not a warning
    print(f"unreadable:{exc.__class__.__name__}")
PY
)
    if [[ "$status" != "complete" ]]; then
      verdict="bad-artifact-status=$status"
    fi
  elif [[ $require_json -eq 1 ]]; then
    verdict=no-artifact
  else
    verdict="ok(no-artifact)"
  fi

  names+=("$name")
  rcs+=("$rc")
  if [[ "$verdict" == ok* ]]; then
    verdicts+=("$verdict")
    echo "== arm $name done rc=0 evidence=$verdict log=$log"
  else
    verdicts+=("$verdict")
    echo "== arm $name done rc=$rc evidence=$verdict log=$log"
  fi
done

echo "-- chain $TAG summary"
fail=0
for i in "${!names[@]}"; do
  printf '   %-22s rc=%-3s %s\n' "${names[$i]}" "${rcs[$i]}" "${verdicts[$i]}"
  [[ "${verdicts[$i]}" == ok* ]] || fail=1
done

if [[ $fail -ne 0 ]]; then
  echo "CHAIN_FAILED $TAG (an arm did not produce a complete measurement)"
  exit 1
fi
echo "CHAIN_DONE $TAG (${#names[@]} arms measured)"
