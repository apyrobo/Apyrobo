# APYROBO Deployment Guide

This guide covers deploying APYROBO in Docker Compose (local/staging) and Kubernetes (production).

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24+ |
| Docker Compose | v2.20+ |
| kubectl | 1.27+ |
| Kubernetes cluster | 1.27+ |

---

## Quick Start — Docker Compose

### 1. Configure environment

```bash
cp docker/.env.example docker/.env
# Edit docker/.env — set APYROBO_API_KEY to a strong random value
```

### 2. Start the API gateway and workers

```bash
docker compose -f docker/docker-compose.yml up -d
```

This starts:
- `apyrobo-api` — REST gateway on port **8080**
- `apyrobo-worker` × 2 — stateless skill executors

Verify:

```bash
curl http://localhost:8080/health
# {"status":"ok","tasks":0,"robots":0}
```

### 3. Optional: enable Redis state persistence

```bash
docker compose -f docker/docker-compose.yml --profile redis up -d
```

### 4. Optional: enable Prometheus metrics

```bash
docker compose -f docker/docker-compose.yml --profile monitoring up -d
# Prometheus UI: http://localhost:9090
# Metrics endpoint: http://localhost:8080/metrics
```

### 4a. Optional: enable full observability stack (Prometheus + Grafana)

```bash
docker compose -f docker/docker-compose.yml --profile observability up -d
# Prometheus UI:  http://localhost:9090
# Grafana:        http://localhost:3000  (admin / apyrobo)
```

The `observability` profile starts both Prometheus and a pre-configured Grafana instance. The dashboard is provisioned automatically — no manual import needed. It includes panels for:

- **Task throughput** — completed and failed tasks per minute
- **Skill latency** — average execution time per skill
- **Task failure rate** — percentage of tasks that failed over the last 5 minutes
- **Skill retry count** — total retries across all skills
- **Error rate by skill** — which skills are failing and how often

The `monitoring` profile (Prometheus only) still works as before for setups that don't need Grafana.

To stop and clean up volumes:

```bash
docker compose -f docker/docker-compose.yml --profile observability down -v
```

### 5. Scale workers

```bash
docker compose -f docker/docker-compose.yml up -d --scale apyrobo-worker=4
```

### 6. View logs

```bash
docker compose -f docker/docker-compose.yml logs -f apyrobo-api
```

---

## Production — Kubernetes

### 1. Build and push the image

```bash
docker build -f docker/Dockerfile -t your-registry/apyrobo:1.0.0 .
docker push your-registry/apyrobo:1.0.0
```

Update the image reference in `k8s/kustomization.yaml`:

```yaml
images:
  - name: apyrobo
    newName: your-registry/apyrobo
    newTag: "1.0.0"
```

### 2. Set secrets

**Do not commit real secrets to git.** Create the secret with `kubectl`:

```bash
kubectl create namespace apyrobo

kubectl create secret generic apyrobo-secret \
  --from-literal=APYROBO_API_KEY="$(openssl rand -hex 32)" \
  --from-literal=APYROBO_REDIS_URL="redis://your-redis-host:6379/0" \
  -n apyrobo \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. Deploy with Kustomize

```bash
kubectl apply -k k8s/
```

### 4. Verify rollout

```bash
kubectl -n apyrobo rollout status deployment/apyrobo-api
kubectl -n apyrobo rollout status deployment/apyrobo-worker
kubectl -n apyrobo get pods
```

### 5. Check the API

```bash
# Get the LoadBalancer IP (may take ~60 s to provision)
kubectl -n apyrobo get svc apyrobo-api

# Test health
curl http://<EXTERNAL-IP>/health
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APYROBO_API_KEY` | *(required)* | Bearer token for authenticating API requests |
| `APYROBO_LOG_LEVEL` | `INFO` | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `APYROBO_REDIS_URL` | *(empty)* | Redis connection URL for shared state. Empty = in-process SQLite |
| `APYROBO_AUDIT_DB` | `/app/data/audit/audit.db` | Path to the SQLite audit database |
| `APYROBO_WORKER_MODE` | `false` | Set `true` on worker pods to enable worker-only behaviour |

---

## Health Check Endpoints

| Endpoint | Auth required | Description |
|----------|--------------|-------------|
| `GET /health` | No | Liveness + readiness probe. Returns `{"status":"ok","tasks":<n>,"robots":<n>}` |
| `GET /metrics` | No | Prometheus metrics (if observability module is active) |

Kubernetes probes are pre-configured in `k8s/deployment-api.yaml`:
- **Liveness** — `GET /health` every 20 s, 3 failures trigger pod restart
- **Readiness** — `GET /health` every 10 s, 3 failures remove pod from load balancer

---

## Scaling Guidelines

### Horizontal scaling

Workers are stateless and scale freely:

```bash
# Docker Compose
docker compose -f docker/docker-compose.yml up -d --scale apyrobo-worker=8

# Kubernetes — manual
kubectl -n apyrobo scale deployment/apyrobo-worker --replicas=8

# Kubernetes — auto (HPA configured in k8s/hpa.yaml)
# Workers scale between 2–10 replicas at 70% CPU / 80% memory
```

### API replicas

The API deployment defaults to **2 replicas** for high availability. Increase for higher throughput:

```bash
kubectl -n apyrobo scale deployment/apyrobo-api --replicas=4
```

### Resource tuning

Default resource requests (per pod):

| Component | CPU request | CPU limit | Memory request | Memory limit |
|-----------|------------|-----------|----------------|-------------|
| API | 100m | 500m | 256 Mi | 512 Mi |
| Worker | 200m | 1000m | 256 Mi | 1 Gi |

Adjust in `k8s/deployment-api.yaml` and `k8s/deployment-worker.yaml` to match your workload.

### Redis

For production deployments with multiple API replicas, enable Redis to share task and robot state:

```bash
# Docker Compose
docker compose -f docker/docker-compose.yml --profile redis up -d

# Kubernetes — point APYROBO_REDIS_URL at a Redis instance or managed service
```

---

## Upgrading

```bash
# Build new image
docker build -f docker/Dockerfile -t your-registry/apyrobo:1.1.0 .
docker push your-registry/apyrobo:1.1.0

# Update kustomization.yaml newTag, then:
kubectl apply -k k8s/

# Monitor the rolling update
kubectl -n apyrobo rollout status deployment/apyrobo-api
```

To roll back:

```bash
kubectl -n apyrobo rollout undo deployment/apyrobo-api
```
