#!/usr/bin/env bash
# Docker smoke test for the API image (Phase 13).
#
#   bash scripts/docker_smoke.sh            # builds fraud-api, runs it, checks it, cleans up
#   IMAGE=my-tag SKIP_BUILD=1 bash scripts/docker_smoke.sh
#
# Steps: build -> run with API_KEY=test on a random free host port -> wait for /ready ->
# POST a SYNTHETIC transaction to /v1/predict with X-API-Key: test -> expect HTTP 200 and a
# risk_tier field (and 401 without the key). The container is removed on every exit path;
# the script exits non-zero if any check fails. No volume mounts: the model is in the image.
set -euo pipefail

IMAGE="${IMAGE:-fraud-api}"
NAME="fraud-api-smoke-$$-${RANDOM}"
WAIT_SECONDS="${WAIT_SECONDS:-90}"
BODY_FILE="$(mktemp)"

fail() {
  echo "SMOKE FAIL: $*" >&2
  echo "---- container logs (last 40 lines) ----" >&2
  docker logs --tail 40 "$NAME" >&2 2>&1 || true
  exit 1
}

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -f "$BODY_FILE"
}
trap cleanup EXIT

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "==> docker build -t $IMAGE ."
  docker build -t "$IMAGE" .
fi

echo "==> docker run $NAME (API_KEY=test, random host port)"
docker run -d --name "$NAME" -e API_KEY=test -p 127.0.0.1::8000 "$IMAGE" >/dev/null
HOST_PORT="$(docker port "$NAME" 8000/tcp | head -n1 | sed 's/.*://')"
[ -n "$HOST_PORT" ] || fail "could not determine mapped host port"
BASE="http://127.0.0.1:${HOST_PORT}"
echo "    listening on $BASE"

echo "==> waiting up to ${WAIT_SECONDS}s for /ready"
ready=""
for _ in $(seq 1 "$WAIT_SECONDS"); do
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    fail "container exited before becoming ready"
  fi
  if curl -fs "$BASE/ready" 2>/dev/null | grep -q '"model_loaded":true'; then
    ready="yes"
    break
  fi
  sleep 1
done
[ -n "$ready" ] || fail "/ready did not report model_loaded=true within ${WAIT_SECONDS}s"
echo "    ready: $(curl -fs "$BASE/ready")"

# SYNTHETIC transaction (not a real row): Time, Amount and V1..V28 = 0.0
PAYLOAD='{"transaction_id":"synthetic-smoke","Time":3600.0,"Amount":25.0'
for i in $(seq 1 28); do PAYLOAD="${PAYLOAD},\"V${i}\":0.0"; done
PAYLOAD="${PAYLOAD}}"

echo "==> POST /v1/predict with X-API-Key: test"
code="$(curl -s -o "$BODY_FILE" -w '%{http_code}' -X POST "$BASE/v1/predict" \
  -H 'Content-Type: application/json' -H 'X-API-Key: test' -d "$PAYLOAD")"
echo "    HTTP $code: $(cat "$BODY_FILE")"
[ "$code" = "200" ] || fail "/v1/predict returned HTTP $code, expected 200"
grep -q '"risk_tier"' "$BODY_FILE" || fail "response has no risk_tier field"

echo "==> POST /v1/predict without a key (expect 401)"
code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/predict" \
  -H 'Content-Type: application/json' -d "$PAYLOAD")"
[ "$code" = "401" ] || fail "unauthenticated /v1/predict returned HTTP $code, expected 401"

echo "SMOKE PASS ($IMAGE)"
