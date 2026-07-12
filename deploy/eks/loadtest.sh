#!/usr/bin/env bash
# Drive enough traffic at the LoadBalancer to push average pod CPU past the HPA
# 60% target and trigger scale-out from 2 toward 5 replicas.
#
# Uses `hey` (an HTTP load generator). Install on macOS with:
#   brew install hey
#
# Usage:
#   ./loadtest.sh http://<elb-hostname>
# Get the ELB hostname with:
#   kubectl -n chartextract get svc chartextract-api \
#     -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
set -euo pipefail

URL="${1:-}"
if [[ -z "${URL}" ]]; then
  echo "ERROR: pass the LoadBalancer URL as the first argument." >&2
  echo "Example: ./loadtest.sh http://a1b2c3.us-east-1.elb.amazonaws.com" >&2
  exit 1
fi

if ! command -v hey >/dev/null 2>&1; then
  echo "ERROR: 'hey' not found. Install it with: brew install hey" >&2
  exit 1
fi

TARGET="${URL%/}/health"

echo "Load target: ${TARGET}"
echo
echo "In a SECOND terminal, watch the autoscaler react (leave it running):"
echo "  kubectl -n chartextract get hpa chartextract-api -w"
echo
echo "And in a THIRD terminal, watch replicas appear:"
echo "  kubectl -n chartextract get pods -w"
echo
echo "Starting a 3-minute load run at concurrency 100 in 5 seconds..."
sleep 5

# -z 3m: run for 3 minutes (long enough for the HPA 15s sync loop to react and
#        add pods, and to capture the scale-out on video).
# -c 100: 100 concurrent workers. /health is cheap per request, so high
#         concurrency is what pushes a single uvicorn worker's CPU up.
hey -z 3m -c 100 "${TARGET}"

echo
echo "Load finished. The HPA will scale back down to 2 after its 120s"
echo "scale-down stabilization window. Capture 'kubectl get hpa' and"
echo "'kubectl get pods' output NOW while replicas are still elevated."
