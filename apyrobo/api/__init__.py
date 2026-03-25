"""
REST API Gateway — HTTP interface for external systems to submit tasks.

Exposes a FastAPI application with:
    POST /tasks              — submit a task
    GET  /tasks/{id}         — get task status
    GET  /robots             — list registered robots
    POST /robots/{id}/skills/{skill} — execute a skill directly
    GET  /health             — health check
"""

from apyrobo.api.app import create_app

__all__ = ["create_app"]
