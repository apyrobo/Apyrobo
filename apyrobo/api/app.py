"""
REST API Gateway — FastAPI application.

Authentication:
    All mutating endpoints require an X-API-Key header validated against
    the APYROBO_API_KEY environment variable (or a key supplied to
    create_app()).  GET /health is public.

In-memory stores:
    Tasks and robots are kept in plain dicts — no database required.

Usage:
    from apyrobo.api.app import create_app
    app = create_app(api_key="secret")

    # with uvicorn:
    # uvicorn apyrobo.api.app:app
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Header, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class TaskRequest(BaseModel):
        skill: str
        params: dict[str, Any] = {}
        robot_id: str = ""

    class SkillRequest(BaseModel):
        params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(api_key: str | None = None) -> "FastAPI":
    """
    Create and return the FastAPI application.

    Args:
        api_key: If provided, all authenticated endpoints require this key
                 in the X-API-Key header.  Defaults to the APYROBO_API_KEY
                 env variable, or an empty string (no auth).
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi and pydantic are required for the REST API gateway")

    _api_key: str = api_key or os.getenv("APYROBO_API_KEY", "")

    # In-memory state
    _tasks: dict[str, dict[str, Any]] = {}
    _robots: dict[str, dict[str, Any]] = {}

    app = FastAPI(title="Apyrobo API", version="0.4.0")

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------

    def require_auth(x_api_key: str = Header(default="")) -> None:
        if _api_key and x_api_key != _api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key",
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "tasks": len(_tasks), "robots": len(_robots)}

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @app.post("/tasks", status_code=status.HTTP_201_CREATED,
              dependencies=[Depends(require_auth)])
    def submit_task(req: TaskRequest) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "skill": req.skill,
            "params": req.params,
            "robot_id": req.robot_id,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
        }
        _tasks[task_id] = task
        return task

    @app.get("/tasks/{task_id}", dependencies=[Depends(require_auth)])
    def get_task(task_id: str) -> dict[str, Any]:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
        return task

    # ------------------------------------------------------------------
    # Robots
    # ------------------------------------------------------------------

    @app.get("/robots", dependencies=[Depends(require_auth)])
    def list_robots() -> list[dict[str, Any]]:
        return list(_robots.values())

    @app.post("/robots/{robot_id}/skills/{skill}",
              dependencies=[Depends(require_auth)])
    def execute_skill(robot_id: str, skill: str, req: SkillRequest) -> dict[str, Any]:
        # Register the robot if not yet seen
        if robot_id not in _robots:
            _robots[robot_id] = {
                "robot_id": robot_id,
                "registered_at": time.time(),
                "capabilities": [],
            }

        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "skill": skill,
            "params": req.params,
            "robot_id": robot_id,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
        }
        _tasks[task_id] = task
        return {"task_id": task_id, "status": "queued"}

    # Expose stores for testing
    app.state.tasks = _tasks
    app.state.robots = _robots

    return app


# ---------------------------------------------------------------------------
# Default app instance (for uvicorn apyrobo.api.app:app)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    app = create_app()
