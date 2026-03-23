"""Tests for apyrobo.skills.feedback"""
import pytest
from apyrobo.skills.feedback import ExecutionResult, FeedbackCollector, AdaptiveExecutor


class TestExecutionResult:
    def test_success_result(self):
        r = ExecutionResult("nav", True, 120.0, None, output="done")
        assert r.skill_name == "nav"
        assert r.success is True
        assert r.duration_ms == 120.0
        assert r.error is None
        assert r.output == "done"

    def test_failure_result(self):
        r = ExecutionResult("pick", False, 50.0, "gripper error")
        assert r.success is False
        assert r.error == "gripper error"


class TestFeedbackCollector:
    def setup_method(self):
        self.fc = FeedbackCollector()

    def test_record_and_success_rate_all_success(self):
        self.fc.record(ExecutionResult("nav", True, 100.0, None))
        self.fc.record(ExecutionResult("nav", True, 110.0, None))
        assert self.fc.success_rate("nav") == 1.0

    def test_success_rate_mixed(self):
        self.fc.record(ExecutionResult("nav", True, 100.0, None))
        self.fc.record(ExecutionResult("nav", False, 50.0, "err"))
        assert self.fc.success_rate("nav") == 0.5

    def test_success_rate_no_data(self):
        # no data = assume healthy
        assert self.fc.success_rate("unknown_skill") == 1.0

    def test_degraded_skills(self):
        for _ in range(3):
            self.fc.record(ExecutionResult("bad_skill", False, 10.0, "err"))
        self.fc.record(ExecutionResult("good_skill", True, 10.0, None))
        degraded = self.fc.degraded_skills(threshold=0.8)
        assert "bad_skill" in degraded
        assert "good_skill" not in degraded

    def test_summary_structure(self):
        self.fc.record(ExecutionResult("nav", True, 100.0, None))
        summary = self.fc.summary()
        assert "skills" in summary
        assert "degraded" in summary
        assert "nav" in summary["skills"]
        assert summary["skills"]["nav"]["total"] == 1

    def test_clear(self):
        self.fc.record(ExecutionResult("nav", True, 100.0, None))
        self.fc.clear()
        assert self.fc.success_rate("nav") == 1.0  # no data → 1.0


class TestAdaptiveExecutor:
    def test_successful_execution(self):
        exec = AdaptiveExecutor(retry_delay_ms=0)
        result = exec.execute("nav", {"x": 1.0}, lambda x: f"moved to {x}")
        assert result.success is True
        assert result.skill_name == "nav"

    def test_failed_execution_recorded(self):
        def always_fail():
            raise RuntimeError("boom")

        exec = AdaptiveExecutor(max_retries=2, retry_delay_ms=0)
        result = exec.execute("bad", {}, always_fail)
        assert result.success is False
        assert "boom" in result.error

    def test_retry_on_failure(self):
        counter = {"n": 0}

        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        exec = AdaptiveExecutor(max_retries=3, retry_delay_ms=0)
        result = exec.execute("flaky", {}, flaky)
        assert result.success is True
        assert counter["n"] == 3

    def test_feedback_collector_populated(self):
        exec = AdaptiveExecutor(retry_delay_ms=0)
        exec.execute("nav", {"x": 0.0}, lambda x: "ok")
        assert exec.collector.success_rate("nav") == 1.0

    def test_adaptive_retry_count_decreases_on_high_rate(self):
        exec = AdaptiveExecutor(max_retries=3, retry_delay_ms=0)
        # Prime with 10 successes → success_rate = 1.0 → only 1 attempt
        for _ in range(10):
            exec.execute("skill", {}, lambda: "ok")
        # Now only 1 attempt should be made
        assert exec._attempts_for("skill") == 1
