"""
Task Queue — priority-based scheduling with preemption.

Manages a queue of pending tasks, assigns them to robots based on
availability and capability, and supports preemption when high-priority
tasks arrive.

Usage:
    queue = TaskQueue()
    queue.submit("deliver_meds", priority=9, robot_id="tb4")
    queue.submit("patrol", priority=2)
    
    # High-priority task preempts lower ones
    next_task = queue.next()  # → deliver_meds (priority 9)
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class QueuedTaskStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


class QueuedTask:
    """A task in the queue with priority and metadata."""

    def __init__(
        self,
        task_description: str,
        priority: int = 5,
        task_id: str | None = None,
        robot_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        deadline: float | None = None,
        preemptible: bool = True,
        submitted_by: str | None = None,
    ) -> None:
        self.task_id = task_id or uuid.uuid4().hex[:10]
        self.task_description = task_description
        self.priority = priority  # 1 (lowest) to 10 (highest)
        self.robot_id = robot_id  # None = auto-assign
        self.parameters = parameters or {}
        self.deadline = deadline  # unix timestamp, None = no deadline
        self.preemptible = preemptible
        self.submitted_by = submitted_by
        self.status = QueuedTaskStatus.QUEUED
        self.assigned_robot: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None

    @property
    def is_overdue(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    @property
    def wait_time(self) -> float:
        """Seconds since submission."""
        return time.time() - self.created_at

    def __lt__(self, other: QueuedTask) -> bool:
        """Higher priority first. Ties broken by earlier creation time."""
        if self.priority != other.priority:
            return self.priority > other.priority  # higher number = higher priority
        return self.created_at < other.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.task_description,
            "priority": self.priority,
            "status": self.status.value,
            "robot_id": self.robot_id,
            "assigned_robot": self.assigned_robot,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "wait_time": round(self.wait_time, 1),
            "is_overdue": self.is_overdue,
        }

    def __repr__(self) -> str:
        return (
            f"<QueuedTask {self.task_id} pri={self.priority} "
            f"status={self.status.value} desc={self.task_description!r:.30}>"
        )


# Type for queue event listeners
QueueListener = Callable[[str, QueuedTask], None]  # (event_type, task)


class TaskQueue:
    """
    Priority queue with preemption support.

    Tasks are ordered by priority (10 = highest). When a task with
    priority >= 8 arrives and all robots are busy with lower-priority
    tasks, the lowest-priority running task is preempted.
    """

    PREEMPT_THRESHOLD = 8  # tasks with priority >= this can preempt

    def __init__(self) -> None:
        self._heap: list[QueuedTask] = []
        self._all_tasks: dict[str, QueuedTask] = {}
        self._running: dict[str, QueuedTask] = {}  # robot_id -> task
        self._lock = threading.Lock()
        self._listeners: list[QueueListener] = []

    def on_event(self, listener: QueueListener) -> None:
        """Register listener for queue events (submitted, assigned, completed, preempted)."""
        self._listeners.append(listener)

    def _emit(self, event_type: str, task: QueuedTask) -> None:
        for listener in self._listeners:
            try:
                listener(event_type, task)
            except Exception as e:
                logger.warning("Queue listener error: %s", e)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, task_description: str, priority: int = 5,
               task_id: str | None = None, robot_id: str | None = None,
               deadline: float | None = None, preemptible: bool = True,
               submitted_by: str | None = None,
               **parameters: Any) -> QueuedTask:
        """Submit a task to the queue."""
        task = QueuedTask(
            task_description=task_description, priority=priority,
            task_id=task_id, robot_id=robot_id, parameters=parameters,
            deadline=deadline, preemptible=preemptible, submitted_by=submitted_by,
        )
        with self._lock:
            heapq.heappush(self._heap, task)
            self._all_tasks[task.task_id] = task
        self._emit("submitted", task)
        logger.info("Queue: submitted %s (pri=%d)", task.task_id, priority)

        # Check if we should preempt a running task
        if priority >= self.PREEMPT_THRESHOLD:
            self._check_preemption(task)

        return task

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def next(self, robot_id: str | None = None) -> QueuedTask | None:
        """
        Get the next task for a robot.

        If robot_id is specified, only returns tasks assigned to (or unassigned for) that robot.
        """
        with self._lock:
            # Rebuild heap to skip cancelled/completed
            candidates = []
            while self._heap:
                task = heapq.heappop(self._heap)
                if task.status != QueuedTaskStatus.QUEUED:
                    continue
                if robot_id and task.robot_id and task.robot_id != robot_id:
                    candidates.append(task)
                    continue
                # Found a match
                task.status = QueuedTaskStatus.ASSIGNED
                task.assigned_robot = robot_id
                # Put remaining back
                for c in candidates:
                    heapq.heappush(self._heap, c)
                self._emit("assigned", task)
                return task
            # Nothing found — put candidates back
            for c in candidates:
                heapq.heappush(self._heap, c)
        return None

    def mark_running(self, task_id: str, robot_id: str) -> None:
        """Mark a task as actively running on a robot."""
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task:
                task.status = QueuedTaskStatus.RUNNING
                task.assigned_robot = robot_id
                task.started_at = time.time()
                self._running[robot_id] = task
                self._emit("running", task)

    def mark_completed(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        """Mark a task as completed."""
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task:
                task.status = QueuedTaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                if task.assigned_robot:
                    self._running.pop(task.assigned_robot, None)
                self._emit("completed", task)

    def mark_failed(self, task_id: str, error: str = "") -> None:
        """Mark a task as failed."""
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task:
                task.status = QueuedTaskStatus.FAILED
                task.completed_at = time.time()
                task.error = error
                if task.assigned_robot:
                    self._running.pop(task.assigned_robot, None)
                self._emit("failed", task)

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task. Cannot cancel running tasks."""
        with self._lock:
            task = self._all_tasks.get(task_id)
            if task and task.status in (QueuedTaskStatus.QUEUED, QueuedTaskStatus.ASSIGNED):
                task.status = QueuedTaskStatus.CANCELLED
                self._emit("cancelled", task)
                return True
        return False

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def _check_preemption(self, new_task: QueuedTask) -> None:
        """Check if the new high-priority task should preempt a running task."""
        with self._lock:
            if not self._running:
                return

            # Find the lowest-priority running task that's preemptible
            lowest: QueuedTask | None = None
            lowest_robot: str | None = None
            for robot_id, running_task in self._running.items():
                if not running_task.preemptible:
                    continue
                if running_task.priority >= new_task.priority:
                    continue
                if lowest is None or running_task.priority < lowest.priority:
                    lowest = running_task
                    lowest_robot = robot_id

            if lowest and lowest_robot:
                logger.warning(
                    "Queue: PREEMPTING task %s (pri=%d) on %s for task %s (pri=%d)",
                    lowest.task_id, lowest.priority, lowest_robot,
                    new_task.task_id, new_task.priority,
                )
                lowest.status = QueuedTaskStatus.PREEMPTED
                self._running.pop(lowest_robot, None)
                self._emit("preempted", lowest)
                # Re-queue the preempted task
                lowest_requeue = QueuedTask(
                    task_description=lowest.task_description,
                    priority=lowest.priority,
                    task_id=f"{lowest.task_id}_requeued",
                    robot_id=lowest.robot_id,
                    parameters=lowest.parameters,
                    preemptible=lowest.preemptible,
                )
                heapq.heappush(self._heap, lowest_requeue)
                self._all_tasks[lowest_requeue.task_id] = lowest_requeue

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self._all_tasks.values() if t.status == QueuedTaskStatus.QUEUED)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def running_tasks(self) -> dict[str, QueuedTask]:
        return dict(self._running)

    def get_task(self, task_id: str) -> QueuedTask | None:
        return self._all_tasks.get(task_id)

    def all_tasks(self) -> list[QueuedTask]:
        return sorted(self._all_tasks.values(), key=lambda t: t.created_at, reverse=True)

    def overdue_tasks(self) -> list[QueuedTask]:
        return [t for t in self._all_tasks.values()
                if t.is_overdue and t.status in (QueuedTaskStatus.QUEUED, QueuedTaskStatus.ASSIGNED)]

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self._all_tasks.values():
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._all_tasks)

    def __repr__(self) -> str:
        return f"<TaskQueue pending={self.pending_count} running={self.running_count} total={len(self)}>"
