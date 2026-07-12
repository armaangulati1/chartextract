#!/usr/bin/env bash
# Build the ChartExtractor image for linux/amd64 (Apple Silicon hosts default to
# arm64, which will not run on EKS x86 nodes), push it to ECR, and write a
# ready-to-apply manifests/deployment.generated.yaml with the pushed image URI.
#
# Required env vars:
#   AWS_ACCOUNT_ID  your 12-digit AWS account id
# Optional env vars:
#   AWS_REGION      defaults to us-east-1
#   IMAGE_TAG       defaults to a short git sha (or "latest" outside a repo)
#   ECR_REPO        defaults to chartextract-api
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPO:-chartextract-api}"

if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
  echo "ERROR: AWS_ACCOUNT_ID is not set." >&2
  echo "Fix: export AWS_ACCOUNT_ID=123456789012   (your 12-digit account id)" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install it first (see RUNBOOK.md prerequisites)." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Start Docker Desktop and retry." >&2
  exit 1
fi

# Resolve a tag: short git sha when available, else "latest".
if [[ -z "${IMAGE_TAG:-}" ]]; then
  if git rev-parse --short HEAD >/dev/null 2>&1; then
    IMAGE_TAG="$(git rev-parse --short HEAD)"
  else
    IMAGE_TAG="latest"
  fi
fi

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}:${IMAGE_TAG}"

# The Dockerfile lives at the repo root; this script lives in deploy/eks. Build
# context must be the repo root so COPY finds api.py et al.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Region:     ${REGION}"
echo "Repository: ${REPO}"
echo "Image URI:  ${IMAGE_URI}"
echo "Context:    ${REPO_ROOT}"
echo

# 1. Create the ECR repo if it does not already exist.
if aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1; then
  echo "ECR repo '${REPO}' already exists. Reusing."
else
  echo "Creating ECR repo '${REPO}'..."
  aws ecr create-repository \
    --repository-name "${REPO}" \
    --region "${REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --tags Key=project,Value=chartextract Key=lifecycle,Value=ephemeral >/dev/null
fi

# 2. Authenticate docker to ECR.
echo "Logging docker in to ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

# 3. Build for linux/amd64 and push in one step via buildx.
echo "Building for linux/amd64 and pushing..."
docker buildx build \
  --platform linux/amd64 \
  --tag "${IMAGE_URI}" \
  --push \
  "${REPO_ROOT}"

# 4. Render the deployment manifest with the real image URI. The generated file
#    is gitignored and is what you kubectl apply.
GENERATED="${SCRIPT_DIR}/manifests/deployment.generated.yaml"
sed "s#IMAGE_PLACEHOLDER#${IMAGE_URI}#g" \
  "${SCRIPT_DIR}/manifests/deployment.yaml" > "${GENERATED}"

echo
echo "Pushed: ${IMAGE_URI}"
echo "Wrote:  ${GENERATED}"
echo "Next:   kubectl apply -f ${GENERATED}"
