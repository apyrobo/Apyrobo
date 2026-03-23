"""
Comprehensive tests for task_queue.py — TaskQueue and QueuedTask.
"""
from __future__ import annotations

import time

import pytest

from apyrobo.task_queue import QueuedTask, QueuedTaskStatus, TaskQueue


# ---------------------------------------------------------------------------
# QueuedTask
# ---------------------------------------------------------------------------

class TestQueuedTask:
    def test_defaults(self) -> None:
        task = QueuedTask("deliver")
        assert task.task_description == "deliver"
        assert task.priority == 5
        assert task.status == QueuedTaskStatus.QUEUED
        assert task.robot_id is None
        assert task.parameters == {}
        assert task.deadline is None
        assert task.preemptible is True
        assert task.submitted_by is None

    def test_custom_id(self) -> None:
        task = QueuedTask("patrol", task_id="abc123")
        assert task.task_id == "abc123"

    def test_auto_id_generated(self) -> None:
        task = QueuedTask("patrol")
        assert len(task.task_id) == 10

    def test_is_overdue_no_deadline(self) -> None:
        task = QueuedTask("scan")
        assert task.is_overdue is False

    def test_is_overdue_past(self) -> None:
        task = QueuedTask("scan", deadline=time.time() - 10)
        assert task.is_overdue is True

    def test_is_overdue_future(self) -> None:
        task = QueuedTask("scan", deadline=time.time() + 3600)
        assert task.is_overdue is False

    def test_wait_time(self) -> None:
        task = QueuedTask("deliver")
        assert task.wait_time >= 0.0

    def test_lt_by_priority(self) -> None:
        high = QueuedTask("h", priority=9)
        low = QueuedTask("l", priority=2)
        assert high < low  # higher priority = "less" in min-heap → pops first

    def test_lt_tiebreak_by_time(self) -> None:
        first = QueuedTask("a", priority=5)
        time.sleep(0.01)
        second = QueuedTask("b", priority=5)
        assert first < second  # earlier creation time wins

    def test_to_dict(self) -> None:
        task = QueuedTask("navigate", priority=7, robot_id="tb4")
        d = task.to_dict()
        assert d["priority"] == 7
        assert d["description"] == "navigate"
        assert d["status"] == "queued"
        assert d["robot_id"] == "tb4"
        assert "task_id" in d
        assert "created_at" in d
        assert "wait_time" in d
        assert "is_overdue" in d

    def test_repr(self) -> None:
        task = QueuedTask("test_task", priority=3)
        r = repr(task)
        assert "QueuedTask" in r
        assert "pri=3" in r
        assert "queued" in r

    def test_parameters(self) -> None:
        task = QueuedTask("deliver", parameters={"dest": "room_A"})
        assert task.parameters == {"dest": "room_A"}


# ---------------------------------------------------------------------------
# TaskQueue — basic operations
# ---------------------------------------------------------------------------

class TestTaskQueueBasic:
    def test_empty_queue(self) -> None:
        q = TaskQueue()
        assert len(q) == 0
        assert q.pending_count == 0
        assert q.running_count == 0

    def test_submit_returns_task(self) -> None:
        q = TaskQueue()
        task = q.submit("deliver")
        assert isinstance(task, QueuedTask)
        assert task.task_description == "deliver"

    def test_submit_increments_count(self) -> None:
        q = TaskQueue()
        q.submit("a")
        q.submit("b")
        assert len(q) == 2
        assert q.pending_count == 2

    def test_next_returns_highest_priority(self) -> None:
        q = TaskQueue()
        q.submit("low", priority=2)
        q.submit("high", priority=8)
        q.submit("mid", priority=5)
        task = q.next()
        assert task is not None
        assert task.task_description == "high"

    def test_next_empty_returns_none(self) -> None:
        q = TaskQueue()
        assert q.next() is None

    def test_next_assigns_status(self) -> None:
        q = TaskQueue()
        q.submit("task")
        task = q.next()
        assert task.status == QueuedTaskStatus.ASSIGNED

    def test_next_with_robot_filter(self) -> None:
        q = TaskQueue()
        q.submit("for_tb4", robot_id="tb4")
        q.submit("for_all")
        # tb3 should only get tasks assigned to tb3 or unassigned
        task = q.next(robot_id="tb3")
        assert task is not None
        assert task.task_description == "for_all"

    def test_next_skips_wrong_robot(self) -> None:
        q = TaskQueue()
        q.submit("for_tb4", robot_id="tb4")
        task = q.next(robot_id="tb3")
        assert task is None  # only task is for tb4

    def test_repr(self) -> None:
        q = TaskQueue()
        q.submit("x")
        r = repr(q)
        assert "TaskQueue" in r
        assert "pending=1" in r

    def test_running_tasks_property(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        running = q.running_tasks
        assert "tb4" in running
        assert running["tb4"].task_id == t.task_id


# ---------------------------------------------------------------------------
# TaskQueue — lifecycle: mark_running / completed / failed / cancel
# ---------------------------------------------------------------------------

class TestTaskQueueLifecycle:
    def test_mark_running(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        assert t.status == QueuedTaskStatus.RUNNING
        assert t.started_at is not None
        assert q.running_count == 1

    def test_mark_completed(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.mark_completed(t.task_id, result={"ok": True})
        assert t.status == QueuedTaskStatus.COMPLETED
        assert t.completed_at is not None
        assert t.result == {"ok": True}
        assert q.running_count == 0

    def test_mark_failed(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.mark_failed(t.task_id, error="timeout")
        assert t.status == QueuedTaskStatus.FAILED
        assert t.error == "timeout"
        assert q.running_count == 0

    def test_cancel_queued(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        # Reset to queued first
        q.submit("cancel_me", task_id="to_cancel")
        result = q.cancel("to_cancel")
        assert result is True
        task = q.get_task("to_cancel")
        assert task.status == QueuedTaskStatus.CANCELLED

    def test_cancel_returns_false_if_not_found(self) -> None:
        q = TaskQueue()
        assert q.cancel("nonexistent") is False

    def test_cancel_running_fails(self) -> None:
        q = TaskQueue()
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        result = q.cancel(t.task_id)
        assert result is False  # can't cancel running task

    def test_mark_completed_unknown_task(self) -> None:
        q = TaskQueue()
        # Should not raise
        q.mark_completed("nonexistent", result=None)

    def test_mark_failed_unknown_task(self) -> None:
        q = TaskQueue()
        q.mark_failed("nonexistent", error="oops")  # should not raise

    def test_mark_running_unknown_task(self) -> None:
        q = TaskQueue()
        q.mark_running("nonexistent", "tb4")  # should not raise


# ---------------------------------------------------------------------------
# TaskQueue — queries
# ---------------------------------------------------------------------------

class TestTaskQueueQueries:
    def test_get_task(self) -> None:
        q = TaskQueue()
        t = q.submit("find_me", task_id="findable")
        found = q.get_task("findable")
        assert found is t

    def test_get_task_missing(self) -> None:
        q = TaskQueue()
        assert q.get_task("missing") is None

    def test_all_tasks(self) -> None:
        q = TaskQueue()
        q.submit("a")
        q.submit("b")
        q.submit("c")
        tasks = q.all_tasks()
        assert len(tasks) == 3

    def test_stats(self) -> None:
        q = TaskQueue()
        q.submit("task1")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.submit("task2")
        s = q.stats()
        assert "running" in s
        assert "queued" in s
        assert s["running"] == 1
        assert s["queued"] == 1

    def test_overdue_tasks(self) -> None:
        q = TaskQueue()
        q.submit("overdue", deadline=time.time() - 1)
        q.submit("fresh", deadline=time.time() + 3600)
        overdue = q.overdue_tasks()
        assert len(overdue) == 1
        assert overdue[0].task_description == "overdue"

    def test_overdue_excludes_completed(self) -> None:
        q = TaskQueue()
        q.submit("task", deadline=time.time() - 1)
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.mark_completed(t.task_id)
        assert q.overdue_tasks() == []


# ---------------------------------------------------------------------------
# TaskQueue — event listeners
# ---------------------------------------------------------------------------

class TestTaskQueueListeners:
    def test_listener_called_on_submit(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("task")
        assert "submitted" in events

    def test_listener_called_on_assign(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("task")
        q.next()
        assert "assigned" in events

    def test_listener_called_on_completed(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.mark_completed(t.task_id)
        assert "completed" in events

    def test_listener_called_on_failed(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("task")
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        q.mark_failed(t.task_id)
        assert "failed" in events

    def test_listener_called_on_cancel(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("task", task_id="cid")
        q.cancel("cid")
        assert "cancelled" in events

    def test_listener_exception_does_not_break_queue(self) -> None:
        q = TaskQueue()

        def bad_listener(evt, t):
            raise RuntimeError("listener crash")

        q.on_event(bad_listener)
        task = q.submit("task")  # should not raise
        assert task is not None


# ---------------------------------------------------------------------------
# TaskQueue — preemption
# ---------------------------------------------------------------------------

class TestTaskQueuePreemption:
    def test_preemption_triggers_on_high_priority(self) -> None:
        q = TaskQueue()
        # Submit and start a low-priority task
        low = q.submit("low", priority=3)
        t = q.next()
        q.mark_running(t.task_id, "tb4")

        preempted_events = []
        q.on_event(lambda evt, task: preempted_events.append(evt) if evt == "preempted" else None)

        # High priority should preempt
        q.submit("urgent", priority=9)
        assert "preempted" in preempted_events

    def test_preemption_requeues_preempted_task(self) -> None:
        q = TaskQueue()
        low = q.submit("low_pri", priority=2)
        t = q.next()
        q.mark_running(t.task_id, "robot1")
        # Submit high-priority — triggers preemption
        q.submit("high_pri", priority=9)
        # The preempted task should be back in queue
        requeued = [t for t in q.all_tasks() if "requeued" in t.task_id]
        assert len(requeued) > 0

    def test_no_preemption_when_none_running(self) -> None:
        q = TaskQueue()
        events = []
        q.on_event(lambda evt, t: events.append(evt))
        q.submit("high", priority=9)  # nothing running to preempt
        assert "preempted" not in events

    def test_non_preemptible_task_not_preempted(self) -> None:
        q = TaskQueue()
        q.submit("protected", priority=2, preemptible=False)
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        events = []
        q.on_event(lambda evt, task: events.append(evt))
        q.submit("urgent", priority=9)
        assert "preempted" not in events

    def test_preemption_threshold_not_triggered_below(self) -> None:
        q = TaskQueue()
        q.submit("low", priority=2)
        t = q.next()
        q.mark_running(t.task_id, "tb4")
        events = []
        q.on_event(lambda evt, task: events.append(evt))
        # Priority 7 is below PREEMPT_THRESHOLD=8
        q.submit("medium", priority=7)
        assert "preempted" not in events
