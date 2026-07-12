<!--
DRAFT ONLY. Do not merge into the main README until the EKS run has actually
been executed and evidence captured in deploy/eks/evidence/. Replace the
evidence placeholders below with the real screenshots before merging.
-->

## Kubernetes (EKS) deployment

The same containerized ChartExtractor API that runs on Render was also deployed
to a Kubernetes cluster on AWS (EKS) to demonstrate container orchestration:
health-gated rollouts, horizontal autoscaling, and self-healing.

### Architecture

```
        Internet
           |
   AWS Network Load Balancer  (Service type: LoadBalancer, port 80)
           |
   EKS cluster: chartextract-demo (us-east-1, 2 x t3.small nodes)
           |
   Deployment: chartextract-api
     - 2 to 5 replicas (HorizontalPodAutoscaler, CPU 60% target)
     - startup + readiness + liveness probes on GET /health (port 8000)
     - credentials injected from a Kubernetes Secret
           |
   Image pulled from Amazon ECR (built for linux/amd64)
```

The image is the repo's existing `Dockerfile` (python:3.11-slim, uvicorn on port
8000). Manifests are plain Kubernetes YAML under
[`deploy/eks/manifests/`](deploy/eks/manifests/); the cluster is defined
declaratively in [`deploy/eks/cluster.yaml`](deploy/eks/cluster.yaml) (eksctl).

### What is demonstrated

- **Health-gated traffic and recovery.** Readiness probes keep a pod out of the
  load balancer rotation until `/health` returns 200; liveness probes restart a
  wedged pod. (`evidence/01-pods-running.png`, `evidence/02-health-ok.txt`)
- **Horizontal autoscaling.** Under load from `hey`, the HorizontalPodAutoscaler
  scales the Deployment from 2 to 5 replicas when average CPU crosses 60% of the
  request, using metrics-server. (`evidence/03-hpa-scaleout.png`,
  `evidence/04-pods-scaled.png`)
- **Self-healing.** Deleting a pod shows the ReplicaSet immediately creating a
  replacement to restore the desired replica count.
  (`evidence/05-selfheal.png`)

### Scope and honesty

This was a **time-boxed demo deployment**, run to capture the evidence above and
then torn down. Specifics, stated plainly:

- Not a hosted or long-running service. The public deployment remains Render
  ([chartextract.onrender.com/health](https://chartextract.onrender.com/health)).
- The service operates on synthetic and public data only (no PHI).
- Single AZ, 2-node cluster, no TLS/ingress hardening, no multi-environment
  setup. These are demo-scope choices, not a template for a real service.
- The cluster was deleted after evidence capture; ongoing cost is $0. Recreating
  it is `deploy/eks/cluster.yaml` plus the scripts (see
  [`deploy/eks/RUNBOOK.md`](deploy/eks/RUNBOOK.md)).

### Cost of one full run

| Item | Rate (us-east-1, on-demand) | Notes |
| --- | --- | --- |
| 2 x t3.small nodes | ~$0.0208/hr each | EC2 compute for the node group |
| EKS control plane | ~$0.10/hr | per-cluster charge |
| Network Load Balancer | ~$0.0225/hr + LCU | from the LoadBalancer Service |
| ECR storage | ~$0.10/GB-month | one small image, prorated |
| EBS (2 x 20 GiB gp3) | ~$0.08/GB-month | node root volumes, prorated |

A full run that is created and torn down in one ~90-minute session costs under
$2. The dominant risk is forgetting teardown, so the runbook makes teardown a
mandatory final phase with a billing-verification checklist and a $20 AWS budget
alarm set up front.

### Reproduce

Full step-by-step guide: [`deploy/eks/RUNBOOK.md`](deploy/eks/RUNBOOK.md).
Manifests, cluster config, and lifecycle scripts (`ecr-push.sh`, `loadtest.sh`,
`selfheal-demo.sh`, `teardown.sh`) are all under `deploy/eks/`.
