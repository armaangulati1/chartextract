# ChartExtractor on EKS: execution runbook

This is a time-boxed demo deployment. You spin up a small EKS cluster, deploy
the ChartExtractor API, capture evidence of probes, autoscaling, and
self-healing, then tear everything down so ongoing cost returns to $0. This is
not a hosted or long-running service.

Everything here runs in the macOS **Terminal** app unless a step explicitly says
to use **Chrome** (the browser). Run one numbered step at a time. Each command
lives in its own code block with nothing else in it, so you can copy the whole
block and paste it. After each command there is an "Expected:" line so you know
it worked.

Total time for a full run: about 60 to 90 minutes, most of it waiting for the
cluster to create (about 15 min) and delete (about 10 to 15 min). Estimated AWS
cost for one full run that is torn down the same session: under $2 (2 x t3.small
nodes plus one ELB, billed hourly). Leaving it running is the expensive mistake,
which is why teardown is mandatory (Phase 9) and why you set a budget alarm
first (Phase 0).

---

## Prerequisites

Install these once. Run each line on its own in Terminal.

1. Homebrew must already be installed. Check it.

```
brew --version
```

Expected: a version line like `Homebrew 4.x.x`. If "command not found", install
Homebrew from brew.sh first.

2. Install the AWS CLI.

```
brew install awscli
```

Expected: ends with a summary; `aws --version` then prints a version.

3. Install eksctl.

```
brew install eksctl
```

Expected: `eksctl version` prints a version like `0.x.x`.

4. Install kubectl.

```
brew install kubectl
```

Expected: `kubectl version --client` prints a client version.

5. Install the load generator.

```
brew install hey
```

Expected: `hey` prints usage text when run with no args.

6. Docker Desktop must be installed and running (whale icon in the menu bar).
Confirm the daemon is up.

```
docker info
```

Expected: a block of server info with no "Cannot connect to the Docker daemon"
error.

7. Configure your AWS credentials (you need an IAM user or SSO with permissions
for EKS, EC2, ECR, CloudFormation, and IAM).

```
aws configure
```

Expected: prompts for Access Key, Secret Key, region (enter `us-east-1`), and
output format (enter `json`). Then confirm identity:

```
aws sts get-caller-identity
```

Expected: JSON with your `Account` (12 digits) and `Arn`. Note the account id;
you need it in Phase 1.

---

## Phase 0: set an AWS budget alarm (do this first)

There is no clean one-line CLI for a budget alarm, so use the console. This is
your safety net if you forget to tear down.

1. In **Chrome**, open the Budgets console.

```
https://us-east-1.console.aws.amazon.com/billing/home#/budgets
```

2. Click "Create budget".
3. Choose "Use a template (simplified)", then "Monthly cost budget".
4. Set the budgeted amount to `20` (dollars).
5. Enter your email address for the alert.
6. Click "Create budget".

Expected: the budget appears in the list with a $20 monthly limit. AWS emails
you if forecast or actual spend crosses the threshold. Budget data can lag a few
hours, so treat this as a backstop, not a real-time guard. The real guard is
Phase 9 teardown.

---

## Phase 1: set your session variables

1. Go to the deploy directory.

```
cd /Users/agulati/Documents/ChartExtractor/deploy/eks
```

Expected: no output. `pwd` shows the path ends in `deploy/eks`.

2. Export your AWS account id (replace the digits with your real account id from
Phase 0 step 7).

```
export AWS_ACCOUNT_ID=123456789012
```

Expected: no output. `echo $AWS_ACCOUNT_ID` prints your account id.

3. Export the region.

```
export AWS_REGION=us-east-1
```

Expected: no output.

Keep this same Terminal window for the rest of the run; these variables live
only in this window.

---

## Phase 2: build and push the image to ECR

Time: about 5 to 10 minutes (first cross-platform build is the slowest part).

1. Make the scripts executable.

```
chmod +x ecr-push.sh loadtest.sh selfheal-demo.sh teardown.sh
```

Expected: no output.

2. Build for linux/amd64 and push to ECR.

```
./ecr-push.sh
```

Expected: it creates the ECR repo, logs docker in ("Login Succeeded"), builds,
pushes, and ends with `Pushed: <account>.dkr.ecr.us-east-1.amazonaws.com/chartextract-api:<tag>`
and `Wrote: .../manifests/deployment.generated.yaml`.

---

## Phase 3: create the EKS cluster

Time: about 15 minutes. This is a good coffee break.

1. Create the cluster from the config file.

```
eksctl create cluster -f cluster.yaml
```

Expected: a long stream of CloudFormation progress ending with
`EKS cluster "chartextract-demo" in "us-east-1" region is ready`. eksctl also
points kubectl at the new cluster automatically.

2. Confirm kubectl can see the two nodes.

```
kubectl get nodes
```

Expected: two nodes listed, both `STATUS Ready`, after a minute or two.

---

## Phase 4: install metrics-server (needed for the HPA)

1. Install metrics-server.

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

Expected: several `created` lines (deployment, service, roles, etc.).

2. Wait for it to become available.

```
kubectl -n kube-system rollout status deployment/metrics-server
```

Expected: `deployment "metrics-server" successfully rolled out` within a minute
or two.

3. Confirm metrics are flowing.

```
kubectl top nodes
```

Expected: a table with CPU and memory usage per node. If it says "metrics not
available yet", wait 30 seconds and rerun.

---

## Phase 5: deploy the app

1. Create the namespace.

```
kubectl apply -f manifests/namespace.yaml
```

Expected: `namespace/chartextract created`.

2. Create the Secret from your shell environment. First export the four values
(use your real Render values; placeholder text is fine if you only want /health
to work). Run these four lines one at a time.

```
export OPENAI_API_KEY=sk-REPLACE
```

```
export DATABASE_URL=postgres://REPLACE
```

```
export LANGFUSE_PUBLIC_KEY=pk-REPLACE
```

```
export LANGFUSE_SECRET_KEY=sk-REPLACE
```

Expected: no output for each.

3. Create the Kubernetes Secret from those variables (no key is written to any
file on disk).

```
kubectl -n chartextract create secret generic chartextract-secrets --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" --from-literal=DATABASE_URL="$DATABASE_URL" --from-literal=LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY" --from-literal=LANGFUSE_SECRET_KEY="$LANGFUSE_SECRET_KEY"
```

Expected: `secret/chartextract-secrets created`.

4. Apply the generated Deployment (the one ecr-push.sh wrote with the real image
URI).

```
kubectl apply -f manifests/deployment.generated.yaml
```

Expected: `deployment.apps/chartextract-api created`.

5. Apply the Service.

```
kubectl apply -f manifests/service.yaml
```

Expected: `service/chartextract-api created`.

6. Apply the HPA.

```
kubectl apply -f manifests/hpa.yaml
```

Expected: `horizontalpodautoscaler.autoscaling/chartextract-api created`.

7. Watch the pods come up.

```
kubectl -n chartextract get pods -w
```

Expected: two pods reach `1/1 Running`. Press Ctrl-C to stop watching.
EVIDENCE CHECKPOINT: save this as `evidence/01-pods-running.png` (screenshot)
or capture the text into `evidence/01-pods-running.txt`.

---

## Phase 6: get the URL and verify health

1. Get the LoadBalancer hostname (the ELB takes 2 to 4 minutes to get one).

```
kubectl -n chartextract get svc chartextract-api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

Expected: a hostname like `a1b2...elb.amazonaws.com`. If blank, wait a minute
and rerun.

2. Save it to a variable (paste the hostname in place of the placeholder).

```
export LB=http://PASTE_HOSTNAME_HERE
```

Expected: no output.

3. Hit the health endpoint (DNS for a new ELB can take a couple of minutes; if
you get a resolve error, wait and retry).

```
curl -s $LB/health
```

Expected: `{"status":"ok"}`.
EVIDENCE CHECKPOINT: save the response into `evidence/02-health-ok.txt`.

---

## Phase 7: autoscaling scale-out (HPA evidence)

You need three Terminal windows. Window A is your existing session.

1. In Window A, start the load test.

```
./loadtest.sh $LB
```

Expected: it prints the watch commands, waits 5 seconds, then streams a running
`hey` load summary for 3 minutes.

2. Open a SECOND Terminal window (Window B) and watch the HPA. First re-point it
at the cluster and namespace by running this single command.

```
kubectl -n chartextract get hpa chartextract-api -w
```

Expected: a row that updates over 30 to 90 seconds. The TARGETS column climbs
above `60%` and REPLICAS rises from `2` toward `5`.
EVIDENCE CHECKPOINT: screenshot the moment REPLICAS increases as
`evidence/03-hpa-scaleout.png`.

3. Open a THIRD Terminal window (Window C) and watch pods appear.

```
kubectl -n chartextract get pods -w
```

Expected: new pods appear in `ContainerCreating` then `Running`, up to 5 total.
EVIDENCE CHECKPOINT: screenshot 4 or 5 running pods as
`evidence/04-pods-scaled.png`.

If CPU never crosses 60% (rare, but /health is a cheap endpoint), lower the HPA
target temporarily and rerun the load: `kubectl -n chartextract patch hpa
chartextract-api --type=merge -p '{"spec":{"metrics":[{"type":"Resource","resource":{"name":"cpu","target":{"type":"Utilization","averageUtilization":30}}}]}}'`.
Note this adjustment in your write-up for honesty.

---

## Phase 8: self-healing (ReplicaSet evidence)

1. In any free Terminal window (make sure you are in the deploy/eks directory),
run the self-heal demo.

```
./selfheal-demo.sh
```

Expected: it prints the current pods, deletes one (`pod "..." deleted`), then
shows a replacement pod with a new name and a few-seconds AGE, then streams
until all pods are Ready again. Press Ctrl-C once they are.
EVIDENCE CHECKPOINT: screenshot the before/after pod list as
`evidence/05-selfheal.png`.

---

## Phase 9: teardown (MANDATORY, do not skip)

Leaving the cluster or ELB running is the only way this demo costs real money.
Do this at the end of the session.

Time: about 10 to 15 minutes.

1. Run the teardown script.

```
./teardown.sh
```

Expected: it deletes the Service (releasing the ELB), waits, deletes the cluster
via eksctl (a long CloudFormation stream ending in the stacks being removed),
deletes the ECR repo, and prints console URLs to verify.

2. Verification checklist. Open each URL in **Chrome** and confirm it is empty.
Check every box before you consider the run done.

   - [ ] EC2 instances: `https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#Instances:` shows no `chartextract-demo` instances running.
   - [ ] Load Balancers: `https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#LoadBalancers:` shows none.
   - [ ] EKS clusters: `https://us-east-1.console.aws.amazon.com/eks/home?region=us-east-1#/clusters` shows no `chartextract-demo`.
   - [ ] CloudFormation: `https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1` shows the `eksctl-chartextract-demo-*` stacks deleted (not stuck in DELETE_FAILED).
   - [ ] Billing: `https://console.aws.amazon.com/billing/home#/` shows no growing EKS/EC2/ELB line for today.

3. If any CloudFormation stack is stuck in DELETE_FAILED, open it, read which
resource blocked deletion (usually a leftover ELB or security group from the
Service), delete that resource manually in its console, then retry the stack
delete. Rerun `./teardown.sh`; it is idempotent.

To recreate later: repeat Phase 2 (ecr-push.sh), Phase 3 (eksctl create), and
Phase 5 (apply manifests). Nothing is preserved between runs by design.
