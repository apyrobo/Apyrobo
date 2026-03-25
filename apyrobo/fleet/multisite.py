"""Multi-site federated fleet management for apyrobo."""
from __future__ import annotations
import json, logging, urllib.request, urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class MultiSiteError(Exception):
    pass


@dataclass
class SiteConfig:
    site_id: str
    name: str
    location: str
    api_url: str
    api_key: str = ""
    timezone: str = "UTC"


@dataclass
class SiteStatus:
    site_id: str
    online: bool
    robot_count: int = 0
    active_tasks: int = 0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MultiSiteManager:
    def __init__(self, local_site_id: str) -> None:
        self.local_site_id = local_site_id
        self._sites: dict[str, SiteConfig] = {}
        self._site_statuses: dict[str, SiteStatus] = {}
        self._rr_index: int = 0

    def register_site(self, config: SiteConfig) -> None:
        self._sites[config.site_id] = config
        self._site_statuses[config.site_id] = SiteStatus(site_id=config.site_id, online=False)

    def unregister_site(self, site_id: str) -> None:
        self._sites.pop(site_id, None)
        self._site_statuses.pop(site_id, None)

    def get_site_status(self, site_id: str) -> Optional[SiteStatus]:
        return self._site_statuses.get(site_id)

    def list_sites(self) -> list[SiteConfig]:
        return list(self._sites.values())

    def submit_task_to_site(self, site_id: str, task: dict) -> str:
        if site_id not in self._sites:
            raise MultiSiteError(f"Unknown site: {site_id}")
        config = self._sites[site_id]
        url = f"{config.api_url.rstrip('/')}/tasks"
        body = json.dumps(task).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": config.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode())
                return data.get("task_id", "unknown")
        except Exception as exc:
            raise MultiSiteError(f"Failed to submit task to {site_id}: {exc}") from exc

    def get_task_status(self, site_id: str, task_id: str) -> dict:
        if site_id not in self._sites:
            raise MultiSiteError(f"Unknown site: {site_id}")
        config = self._sites[site_id]
        url = f"{config.api_url.rstrip('/')}/tasks/{urllib.parse.quote(task_id)}"
        req = urllib.request.Request(url, headers={"X-API-Key": config.api_key})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            raise MultiSiteError(f"Failed to get task status from {site_id}: {exc}") from exc

    def route_task(self, task: dict, strategy: str = "least_loaded") -> tuple[str, str]:
        site_id = self._select_site(strategy)
        if site_id is None:
            raise MultiSiteError("No sites available for routing")
        task_id = self.submit_task_to_site(site_id, task)
        return (site_id, task_id)

    def sync_state(self) -> dict:
        for site_id, config in self._sites.items():
            url = f"{config.api_url.rstrip('/')}/status"
            req = urllib.request.Request(url, headers={"X-API-Key": config.api_key})
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode())
                    self._site_statuses[site_id] = SiteStatus(
                        site_id=site_id,
                        online=True,
                        robot_count=data.get("robot_count", 0),
                        active_tasks=data.get("active_tasks", 0),
                    )
            except Exception:
                self._site_statuses[site_id] = SiteStatus(site_id=site_id, online=False)
        return {sid: s.online for sid, s in self._site_statuses.items()}

    def _select_site(self, strategy: str) -> Optional[str]:
        online = [sid for sid, s in self._site_statuses.items() if s.online]
        if not online:
            online = list(self._sites.keys())  # fallback: try all
        if not online:
            return None
        if strategy == "least_loaded":
            return min(online, key=lambda sid: self._site_statuses.get(sid, SiteStatus(sid, False)).active_tasks)
        elif strategy == "closest":
            return online[0]  # simplified: first registered is "closest"
        elif strategy == "round_robin":
            site = online[self._rr_index % len(online)]
            self._rr_index += 1
            return site
        else:
            return online[0]
