"""Orchestration adapter base — pluggable interface for any front-end."""
from apyrobo.orchestration.adapter import (
    OrchestrationAdapter,
    OrchestrationServer,
    StdioOrchestrationAdapter,
    MockOrchestrationAdapter,
    OrchestrationMessage,
    WebSocketOrchestrationAdapter,
)
from apyrobo.orchestration.slack_adapter import SlackOrchestrationAdapter

__all__ = [
    "OrchestrationAdapter",
    "OrchestrationServer",
    "StdioOrchestrationAdapter",
    "MockOrchestrationAdapter",
    "OrchestrationMessage",
    "WebSocketOrchestrationAdapter",
    "SlackOrchestrationAdapter",
]
