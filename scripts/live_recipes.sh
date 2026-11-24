#!/usr/bin/env bash
# Live recipe sweep: Goose sessions (Rust codegen + build loop, MCP, skills) through the bridge.
# Waits for IAedu's rate limit to lift, runs each recipe, verifies the generated crate with
# cargo (independent of the agent's claims) and prints a summary.
# Usage: scripts/live_recipes.sh [recipe-name ...]   (default: all recipes/*.yaml)
# Env:   MAX_WAIT (s, default 5400)  PROBE_EVERY (s, default 300)  RECIPE_TIMEOUT (s, default 1800)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${TMPDIR:-/tmp}/iaedu/live"
mkdir -p "$out"
set -a
. "$root/.env"
set +a
export IAEDU_BRIDGE_API_KEY="${BRIDGE_API_KEY:-unused}"
bridge_log="$out/bridge.log"

probe() { # 0 when IAedu answers with a token instead of a rate-limit error
  curl -sS -m 60 -N -X POST "$IAEDU_ENDPOINT" -H "x-api-key: $IAEDU_API_KEY" \
    -F "channel_id=$IAEDU_CHANNEL_ID" -F "thread_id=probe-$RANDOM" -F 'user_info={}' \
    -F 'message=Reply with: ok' | grep -q '"type": "token", "content": "[^"]'
}

waited=0
until probe; do
  if [ "$waited" -ge "${MAX_WAIT:-5400}" ]; then
    echo "IAedu still rate limited after ${waited}s: giving up" >&2
    exit 2
  fi
  echo "$(date +%T) rate limited, waiting ${PROBE_EVERY:-300}s"
  sleep "${PROBE_EVERY:-300}"
  waited=$((waited + ${PROBE_EVERY:-300}))
done
echo "$(date +%T) IAedu answers: starting sweep"

PYTHONPATH="$root/src" "$root/venv/bin/python" -m iaedu_bridge >"$bridge_log" 2>&1 &
bridge_pid=$!
trap 'kill $bridge_pid 2>/dev/null || true' EXIT
sleep 3

recipes=("$@")
[ ${#recipes[@]} -eq 0 ] && mapfile -t recipes < <(basename -a "$root"/recipes/*.yaml | sed 's/\.yaml$//')

summary="$out/summary.txt"
printf '%-18s %-6s %-7s %-6s %-6s %-6s %s\n' recipe goose secs reqs retry errs cargo >"$summary"
for name in "${recipes[@]}"; do
  work="$out/$name"
  rm -rf "$work"
  mkdir -p "$work"
  before_req=$(grep -c 'chat thread=' "$bridge_log" || true)
  before_retry=$(grep -c ' retry ' "$bridge_log" || true)
  before_err=$(grep -c 'upstream error' "$bridge_log" || true)
  start=$SECONDS
  rc=0
  GOOSE_PROVIDER=custom_iaedu GOOSE_MODEL=iaedu-5.5 timeout "${RECIPE_TIMEOUT:-1800}" \
    goose run --no-session --no-profile --max-turns 80 \
    --recipe "$root/recipes/$name.yaml" --params "workdir=$work/ws" >"$out/$name.goose.log" 2>&1 || rc=$?
  secs=$((SECONDS - start))
  reqs=$(($(grep -c 'chat thread=' "$bridge_log" || true) - before_req))
  retries=$(($(grep -c ' retry ' "$bridge_log" || true) - before_retry))
  errs=$(($(grep -c 'upstream error' "$bridge_log" || true) - before_err))
  # independent verification of the generated crate
  cargo_res="no-crate"
  manifest=$(find "$work/ws" -maxdepth 2 -name Cargo.toml 2>/dev/null | head -1 || true)
  if [ -n "$manifest" ]; then
    dir=$(dirname "$manifest")
    cargo_res="build-fail"
    if (cd "$dir" && cargo build -q >"$out/$name.build.log" 2>&1); then
      cargo_res="test-fail"
      (cd "$dir" && cargo test -q >"$out/$name.test.log" 2>&1) && cargo_res="pass"
      if [ "$name" = rust_expr_eval ] && [ "$cargo_res" = pass ]; then
        (cd "$dir" && cargo clippy -q -- -D warnings >"$out/$name.clippy.log" 2>&1) || cargo_res="pass+clippy-fail"
      fi
    fi
  fi
  printf '%-18s %-6s %-7s %-6s %-6s %-6s %s\n' "$name" "$rc" "${secs}s" "$reqs" "$retries" "$errs" "$cargo_res" >>"$summary"
done
echo
cat "$summary"
echo "logs: $out (bridge.log, <recipe>.goose.log, <recipe>.build.log, <recipe>.test.log)"
