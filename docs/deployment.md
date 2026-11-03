# Deployment examples (UNVERIFIED: Docker and kubectl were not available when these were written)

The files are YAML-parse-checked and the API image's file set was import-checked, but **nothing was built or applied**.
No cloud deployment exists or is claimed; no URL is provided.

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
```
docker build -t aiml01-api:local . ; docker build -t aiml01-web:local ./frontend
minikube start
minikube image load aiml01-api:local ; minikube image load aiml01-web:local
kubectl apply -f deploy/k8s/
# copy a trained run folder into the models volume (e.g. a temporary pod), set MODEL_ARTIFACT_DIR in the ConfigMap, restart the api Deployment
kubectl -n aiml01 port-forward svc/web 8080:8080
```
Limits: single replica by design (SQLite + in-memory rate limiter); readiness stays failing until a valid model is mounted, by design;
no Ingress, TLS, secrets management, autoscaling or authentication; storage is a plain PVC. Treat these as teaching examples.

## CI (`.github/workflows/ci.yml`)
Jobs: Python (`compileall` + `pytest`), optional real-XGBoost test (non-blocking), frontend (typecheck, lib tests, component tests, build),
Docker builds. Never executed; the first push may reveal failures, especially in the frontend component tests and the HTTP tests.
