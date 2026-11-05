# Local Docker Compose and Kubernetes deployment

## Docker Compose (local)
```
docker compose build                       # add --build-arg INSTALL_XGBOOST=true to include xgboost
$env:CONTAINER_MODEL_DIR = "/models/<run_id>/primary"      # a path INSIDE the container; ./ml/artifacts is mounted at /models (read-only)
docker compose up
# API  http://127.0.0.1:8000/api/v1/health    Dashboard  http://127.0.0.1:8080
```
Ports bind to loopback because the API has no authentication. Containers run as non-root; the API's only writable path is the
`api-data` volume at `/data`. **Version caveat:** scikit-learn artifacts are rejected if the container's scikit-learn minor version
differs from the one that trained them; train inside the same environment or use xgboost (native JSON model, major-version check).
`APP_ENV=production` disables `/docs`.

## Kubernetes / Minikube (example only)
## Kubernetes / Minikube (Windows PowerShell)

Minikube runs locally only. The API has no authentication, so do not create an Ingress, LoadBalancer, or NodePort. Use loopback port-forwarding only.

```powershell
Set-Location "C:\Projects\AIML-01-Fraud-Detection"

docker build --build-arg INSTALL_XGBOOST=true -t aiml01-api:local .
docker build -t aiml01-web:local ./frontend
minikube image load aiml01-api:local
minikube image load aiml01-web:local

kubectl apply -f deploy/k8s/00-namespace-config.yaml
kubectl apply -f deploy/k8s/10-storage.yaml
kubectl apply -f deploy/k8s/model-copy-pod.yaml
kubectl apply -f deploy/k8s/20-api.yaml
kubectl apply -f deploy/k8s/30-web.yaml

kubectl get nodes
kubectl get pods -n aiml01
kubectl get pvc -n aiml01
kubectl get services -n aiml01
kubectl get configmaps,secrets -n aiml01

kubectl wait --for=condition=complete job/model-copy -n aiml01 --timeout=180s
kubectl logs job/model-copy -n aiml01
kubectl rollout status deployment/api -n aiml01 --timeout=180s
kubectl rollout status deployment/web -n aiml01 --timeout=180s
```

Architecture:

- `aiml01-data` stores the SQLite database and streaming output state.
- `aiml01-models` stores the complete copied artifact at `/models/pre-balance-full-v2/primary`.
- The `model-copy` Job copytrees the complete primary directory from the API image into the models PVC.
- The API mounts the models PVC read-only and loads `/models/pre-balance-full-v2/primary`.
- The frontend is Nginx and proxies `/api/*` to the in-cluster `api:8000` Service.
- API and frontend containers run with restricted security contexts and no service-account token.

Local access:

```powershell
kubectl -n aiml01 port-forward svc/api 8000:8000
kubectl -n aiml01 port-forward svc/web 8080:8080
```

In a second PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready
Invoke-RestMethod http://127.0.0.1:8000/api/v1/model

$body = @{
  step = 600
  type = "TRANSFER"
  amount = 250000
  nameOrig = "C1234567890"
  nameDest = "C9876543210"
  oldbalanceOrg = 250000
  oldbalanceDest = 0
  newbalanceOrig = 0
  newbalanceDest = 250000
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/predict `
  -ContentType "application/json" `
  -Body $body
```

Troubleshooting:

```powershell
kubectl describe job/model-copy -n aiml01
kubectl logs job/model-copy -n aiml01
kubectl get events -n aiml01 --sort-by=.lastTimestamp
kubectl logs deployment/api -n aiml01
kubectl describe pod <api-pod-name> -n aiml01
```

The API ConfigMap must not be used as the model artifact source. There is no application ConfigMap volume mount; `MODEL_ARTIFACT_DIR` is an environment variable, and the model PVC is mounted at `/models`.

Limits: single replica by design (SQLite + in-process memory rate limiter); readiness stays failing until a valid model is mounted; no Ingress, TLS, secrets management, autoscaling or authentication; storage is a plain RWO PVC.

## CI (`.github/workflows/ci.yml`)
Jobs: Python (`compileall` + `pytest`), optional real-XGBoost test (non-blocking), frontend (typecheck, lib tests, component tests, build),
Docker builds. Never executed; the first push may reveal failures, especially in the frontend component tests and the HTTP tests.
