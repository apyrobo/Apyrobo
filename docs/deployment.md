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

### 2. Start the API gateway

```bash
docker compose -f docker/docker-compose.yml up -d
```

This starts:
- `apyrobo-api` — REST gateway on port **8080** (`apyrobo.api.app`). It
  records submitted tasks and skill calls with status `queued` and returns
  them by id; nothing in it runs them on a robot yet, and the image has no
  ROS 2. To drive a robot, run `apyrobo mcp` or `apyrobo serve --execute`
  where ROS 2 is sourced (see [first-real-robot.md](first-real-robot.md))

Verify:

```bash
curl http://localhost:8080/health
# {"status":"ok","tasks":0,"robots":0}
```

### 3. What the gateway does not do yet

The gateway keeps tasks and robots in memory, per process: a restart forgets
them, and two replicas do not see each other's tasks. It reads one setting,
`APYROBO_API_KEY`, and serves no `/metrics`. The compose file's `redis`,
`monitoring` and `observability` profiles start Redis, Prometheus and Grafana
next to it, but the gateway does not use Redis and Prometheus finds nothing to
scrape, so the Grafana panels stay empty. Shared state and metrics are roadmap
items.

### 4. View logs

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
| `APYROBO_API_KEY` | *(empty: no auth)* | Key every request except `GET /health` must send in the `X-API-Key` header |

The compose file, `docker/.env.example` and the Kubernetes ConfigMap and
Secret also set `APYROBO_LOG_LEVEL`, `APYROBO_REDIS_URL` and
`APYROBO_AUDIT_DB`; the gateway does not read them yet.

---

## Health Check Endpoints

| Endpoint | Auth required | Description |
|----------|--------------|-------------|
| `GET /health` | No | Liveness + readiness probe. Returns `{"status":"ok","tasks":<n>,"robots":<n>}` |

Kubernetes probes are pre-configured in `k8s/deployment-api.yaml`:
- **Liveness** — `GET /health` every 20 s, 3 failures trigger pod restart
- **Readiness** — `GET /health` every 10 s, 3 failures remove pod from load balancer

---

## Scaling Guidelines

### Replicas

The deployment runs **1 replica**. Task and robot state lives in each
process's memory, so with more replicas a `GET /tasks/<id>` that lands on
another pod answers 404. Keep one replica until the gateway has shared state.

> A dedicated stateless worker tier (`apyrobo worker`) was described in
> earlier versions of this guide but was never implemented; the service and
> manifests have been removed. A distributed executor tier is a roadmap
> item.

### Resource tuning

Default resource requests (per pod):

| Component | CPU request | CPU limit | Memory request | Memory limit |
|-----------|------------|-----------|----------------|-------------|
| API | 100m | 500m | 256 Mi | 512 Mi |

Adjust in `k8s/deployment-api.yaml` to match your workload.

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
