"""
Multi-site fleet management — federated control across physical locations.

Supports task routing with least-loaded, closest, and round-robin strategies.
Uses only stdlib (urllib) for HTTP to avoid extra dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class MultiSiteError(Exception):
    """Raised for multi-site fleet management errors."""


@dataclass
class SiteConfig:
    site_id: str
    name: str
    location: str
    api_url: str
    api_key: str
    timezone: str


@dataclass
class SiteStatus:
    site_id: str
    online: bool
    robot_count: int
    active_tasks: int
    last_heartbeat: datetime


class MultiSiteManager:
    """Federated fleet management across multiple physical sites."""

    def __init__(self, local_site_id: str) -> None:
        self.local_site_id = local_site_id
        self._sites: dict[str, SiteConfig] = {}
        self._site_statuses: dict[str, SiteStatus] = {}
        self._round_robin_index: int = 0

    # ------------------------------------------------------------------
    # Site registry
    # ------------------------------------------------------------------

    def register_site(self, config: SiteConfig) -> None:
        self._sites[config.site_id] = config

    def unregister_site(self, site_id: str) -> None:
        self._sites.pop(site_id, None)
        self._site_statuses.pop(site_id, None)

    def get_site_status(self, site_id: str) -> Optional[SiteStatus]:
        return self._site_statuses.get(site_id)

    def list_sites(self) -> list[SiteConfig]:
        return list(self._sites.values())

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit_task_to_site(self, site_id: str, task: dict) -> str:
        """POST task JSON to remote site; return task_id."""
        if site_id not in self._sites:
            raise MultiSiteError(f"Unknown site: {site_id}")
        config = self._sites[site_id]
        url = f"{config.api_url.rstrip('/')}/tasks"
        payload = json.dumps(task).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                return body["task_id"]
        except urllib.error.URLError as exc:
            raise MultiSiteError(f"HTTP error submitting task to {site_id}: {exc}") from exc

    def get_task_status(self, site_id: str, task_id: str) -> dict:
        if site_id not in self._sites:
            raise MultiSiteError(f"Unknown site: {site_id}")
        config = self._sites[site_id]
        url = f"{config.api_url.rstrip('/')}/tasks/{task_id}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise MultiSiteError(f"HTTP error fetching task from {site_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_task(self, task: dict, strategy: str = "least_loaded") -> tuple[str, str]:
        """Route task to a site and return (site_id, task_id)."""
        site_id = self._select_site(strategy)
        if site_id is None:
            raise MultiSiteError("No sites available for routing")
        task_id = self.submit_task_to_site(site_id, task)
        return site_id, task_id

    def _select_site(self, strategy: str) -> Optional[str]:
        site_ids = list(self._sites.keys())
        if not site_ids:
            return None

        if strategy == "least_loaded":
            online = [
                s for s in self._site_statuses.values() if s.online and s.site_id in self._sites
            ]
            if not online:
                return site_ids[0]
            return min(online, key=lambda s: s.active_tasks).site_id

        if strategy == "closest":
            # Without geodata we use list order as a proxy; first registered = "closest"
            return site_ids[0]

        if strategy == "round_robin":
            idx = self._round_robin_index % len(site_ids)
            self._round_robin_index += 1
            return site_ids[idx]

        raise MultiSiteError(f"Unknown routing strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def sync_state(self) -> dict:
        """Poll all registered sites for current status; update internal cache."""
        results: dict[str, Any] = {}
        for site_id, config in self._sites.items():
            url = f"{config.api_url.rstrip('/')}/status"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                status = SiteStatus(
                    site_id=site_id,
                    online=data.get("online", True),
                    robot_count=data.get("robot_count", 0),
                    active_tasks=data.get("active_tasks", 0),
                    last_heartbeat=datetime.fromisoformat(
                        data.get("last_heartbeat", datetime.utcnow().isoformat())
                    ),
                )
                self._site_statuses[site_id] = status
                results[site_id] = status
            except (urllib.error.URLError, KeyError, ValueError):
                status = SiteStatus(
                    site_id=site_id,
                    online=False,
                    robot_count=0,
                    active_tasks=0,
                    last_heartbeat=datetime.utcnow(),
                )
                self._site_statuses[site_id] = status
                results[site_id] = status
        return results
