#!/usr/bin/env bash
# Delete everything this demo created so ongoing spend returns to $0. Idempotent:
# safe to run more than once, and safe to run even if some pieces are already
# gone. Deletes in dependency order so the AWS ELB is released before the
# cluster.
#
# Required env vars (same as ecr-push.sh):
#   AWS_ACCOUNT_ID
# Optional:
#   AWS_REGION  defaults to us-east-1
#   ECR_REPO    defaults to chartextract-api
set -uo pipefail   # not -e: we want to attempt every cleanup step even if one fails

REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPO:-chartextract-api}"
CLUSTER="chartextract-demo"
NS="chartextract"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "This will DELETE:"
echo "  - Kubernetes Service '${NS}/chartextract-api' (releases the AWS ELB)"
echo "  - EKS cluster '${CLUSTER}' (region ${REGION}, includes the node group)"
echo "  - ECR repository '${REPO}' and all its images"
echo

# 1. Delete the LoadBalancer Service first so AWS deprovisions the ELB. Deleting
#    the cluster before the Service can orphan the ELB and keep billing it.
if command -v kubectl >/dev/null 2>&1; then
  echo "Deleting LoadBalancer Service (releasing the ELB)..."
  kubectl -n "${NS}" delete svc chartextract-api --ignore-not-found=true
  echo "Waiting 30s for AWS to deprovision the ELB before deleting the cluster..."
  sleep 30
else
  echo "kubectl not found; skipping Service delete. eksctl cleanup below should"
  echo "still remove the ELB via the VPC, but verify the EC2 > Load Balancers"
  echo "console shows none afterward."
fi

# 2. Delete the cluster (node group + control plane + VPC).
if command -v eksctl >/dev/null 2>&1; then
  echo "Deleting EKS cluster '${CLUSTER}'..."
  eksctl delete cluster --name "${CLUSTER}" --region "${REGION}" --wait
else
  echo "ERROR: eksctl not found; cannot delete the cluster automatically." >&2
  echo "Delete it manually: eksctl delete cluster --name ${CLUSTER} --region ${REGION}" >&2
fi

# 3. Delete the ECR repo (force removes any images).
if command -v aws >/dev/null 2>&1; then
  echo "Deleting ECR repository '${REPO}'..."
  aws ecr delete-repository --repository-name "${REPO}" --region "${REGION}" --force >/dev/null 2>&1 \
    && echo "ECR repo deleted." \
    || echo "ECR repo already absent or could not be deleted; verify in console."
else
  echo "aws CLI not found; delete the ECR repo manually in the console."
fi

# 4. Remove the locally generated manifest so a re-run starts clean.
rm -f "${SCRIPT_DIR}/manifests/deployment.generated.yaml"

echo
echo "Teardown attempted. VERIFY nothing is still billing:"
echo "  - EC2 instances (should be none):      https://console.aws.amazon.com/ec2/home?region=${REGION}#Instances:"
echo "  - Load Balancers (should be none):     https://console.aws.amazon.com/ec2/home?region=${REGION}#LoadBalancers:"
echo "  - EKS clusters (should be none):        https://console.aws.amazon.com/eks/home?region=${REGION}#/clusters"
echo "  - Overall billing dashboard:            https://console.aws.amazon.com/billing/home#/"
echo
echo "To recreate later: run ecr-push.sh, then 'eksctl create cluster -f cluster.yaml',"
echo "then re-apply the manifests (see RUNBOOK.md). No state is preserved between runs."
