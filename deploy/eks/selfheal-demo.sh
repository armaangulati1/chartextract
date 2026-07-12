#!/usr/bin/env bash
# Demonstrate self-healing: delete one pod and watch the Deployment's ReplicaSet
# create a replacement automatically, so the replica count returns to its floor.
set -euo pipefail

NS="chartextract"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "ERROR: kubectl not found." >&2
  exit 1
fi

echo "Current pods:"
# Expected: 2 (or more, if a load test scaled out) pods, all Running.
kubectl -n "${NS}" get pods -l app=chartextract -o wide

VICTIM="$(kubectl -n "${NS}" get pods -l app=chartextract \
  -o jsonpath='{.items[0].metadata.name}')"

if [[ -z "${VICTIM}" ]]; then
  echo "ERROR: no chartextract pods found. Is the Deployment applied?" >&2
  exit 1
fi

echo
echo "Deleting one pod to simulate a crash: ${VICTIM}"
# Expected: "pod \"<name>\" deleted". The ReplicaSet immediately notices the
# replica count dropped below the desired count and creates a new pod.
kubectl -n "${NS}" delete pod "${VICTIM}"

echo
echo "Pods within a few seconds of the delete:"
# Expected: a brand-new pod in ContainerCreating or Running with a DIFFERENT
# name and an AGE of a few seconds, alongside the survivors.
kubectl -n "${NS}" get pods -l app=chartextract -o wide

echo
echo "Watching until all pods are Ready again (Ctrl-C once they are):"
echo "Expected: replacement pod reaches Running / 1-1 READY, total back to floor."
# Blocks and streams changes. The final steady state matches the pre-delete
# count because the Deployment reconciles the missing replica.
kubectl -n "${NS}" get pods -l app=chartextract -w
