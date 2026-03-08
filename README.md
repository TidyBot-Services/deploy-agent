# Deploy Agent

Lightweight service deployment daemon for compute nodes. Runs on each GPU server and manages Docker containers for ML/vision services. AI agents (OpenClaw/skill agents) call this to deploy, discover, and stop services.

**GitHub:** [TidyBot-Services/deploy-agent](https://github.com/TidyBot-Services/deploy-agent)

## How It Fits

```
AI Agent (OpenClaw)
    │
    ├── GET  /services      →  What's running?
    ├── POST /deploy        →  Start a service container
    ├── POST /stop          →  Stop a service
    └── GET  /gpus          →  GPU availability
    │
Deploy Agent (:9000)
    │
    └── Docker containers (ports 8000-8999)
        ├── yolo-service        :8010
        ├── grounded-sam2       :8001
        ├── contact-graspnet    :8011
        └── ...
```

Each compute server runs one deploy agent. The AI agent is the orchestrator — it decides what to deploy and where, based on GPU availability and service needs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (requires Docker access)
python server.py --port 9000
```

Requires Docker socket access (`/var/run/docker.sock`).

## API Endpoints

### `GET /health`

Server health and summary.

```json
{
  "status": "ok",
  "hostname": "gpu-server-1",
  "gpus": 2,
  "services_running": 3
}
```

### `GET /services`

List all running services with health status.

```json
[
  {
    "name": "yolo",
    "host": "http://158.130.109.188:8010",
    "port": 8010,
    "gpu": 0,
    "status": "healthy",
    "image": "tidybot/yolo-service"
  }
]
```

### `GET /services/{name}`

Get details for a specific service (404 if not deployed).

### `POST /deploy`

Deploy a service container. If already running and healthy, returns the existing instance.

```json
{
  "name": "yolo",
  "image": "tidybot/yolo-service:latest",
  "port": 8000,
  "gpu": true,
  "vram_gb": 4,
  "env": {"MODEL_SIZE": "large"},
  "volumes": ["/data/models:/models"],
  "health": "/health",
  "ready_timeout": 60,
  "command": null
}
```

**Response:**
```json
{
  "ok": true,
  "name": "yolo",
  "host": "http://158.130.109.188:8010",
  "gpu": 0,
  "status": "healthy",
  "already_running": false
}
```

**Behavior:**
- Picks GPU with most free VRAM (if `gpu: true`)
- Assigns preferred port or next available in 8000-8999 range
- Pulls image if not available locally
- Waits for health check before returning
- Returns 503 if health check times out (container is cleaned up)
- Idempotent: if service is already healthy, returns existing instance

### `POST /stop`

Stop and remove a service container.

```json
{"name": "yolo"}
```

### `GET /gpus`

GPU status with VRAM usage and assigned services.

```json
[
  {
    "id": 0,
    "name": "NVIDIA RTX 4090",
    "vram_total_gb": 24.0,
    "vram_used_gb": 8.2,
    "services": ["yolo", "grounded-sam2"]
  }
]
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `9000` | Listen port |
| `DEPLOY_STATE_DIR` | `/var/lib/deploy-agent` | Persistent state directory |

Port range for services: **8000–8999** (9000 is reserved for the deploy agent itself).

## State Persistence

Running services are saved to `$DEPLOY_STATE_DIR/services.json`. On restart, the deploy agent reconciles with Docker — if a container is still running, it's re-adopted; if gone, it's dropped from state.
