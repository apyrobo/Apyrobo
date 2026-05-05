"""Sim-to-real transfer — validate plans in simulation before real deployment.

Runs a plan against a simulation adapter (e.g. Gazebo) and, optionally,
deploys to the real robot only if the simulation succeeds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Outcome of running a plan against a simulator."""

    success: bool
    steps_completed: int
    steps_total: int
    failures: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    robot_final_position: dict[str, float] = field(default_factory=dict)


class SimToRealTransfer:
    """Validate a plan in simulation, then optionally deploy to the real robot.

    Args:
        sim_adapter_uri:  URI of the simulation robot (e.g. ``"gazebo://robot"``).
        real_adapter_uri: URI of the real robot.  May be ``None`` if deployment
                          is not needed.
    """

    def __init__(
        self,
        sim_adapter_uri: str = "gazebo://robot",
        real_adapter_uri: str | None = None,
    ) -> None:
        self._sim_uri = sim_adapter_uri
        self._real_uri = real_adapter_uri

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, plan: Any, agent: Any = None, timeout_s: float = 60.0) -> SimulationResult:
        """Run *plan* against the simulation adapter and return the result.

        *plan* may be a :class:`~apyrobo.skills.executor.SkillGraph` or any
        object understood by the sim adapter.  If ``agent`` is provided, it is
        used to execute the plan on the sim robot; otherwise a basic stub
        execution is performed.

        Args:
            plan:      The skill graph or task string to validate.
            agent:     Optional agent for execution.
            timeout_s: Maximum simulation wall-clock seconds.

        Returns:
            :class:`SimulationResult` capturing outcome.
        """
        try:
            from apyrobo.core.robot import Robot
            from apyrobo.core.schemas import TaskStatus
            sim_robot = Robot.discover(self._sim_uri)

            if agent is not None:
                task = str(plan) if not hasattr(plan, "get_execution_order") else "sim_task"
                if hasattr(plan, "get_execution_order"):
                    from apyrobo.skills.executor import SkillExecutor
                    executor = SkillExecutor(sim_robot)
                    result = executor.execute_graph(plan)
                else:
                    result = agent.execute(str(plan), sim_robot)

                status_val = (
                    result.status.value
                    if hasattr(result.status, "value")
                    else str(result.status)
                )
                success = status_val == "completed"
                failures = [result.error] if result.error and not success else []
                return SimulationResult(
                    success=success,
                    steps_completed=result.steps_completed,
                    steps_total=result.steps_total,
                    failures=failures,
                    robot_final_position={},
                )
            else:
                # Basic stub: execute plan steps against sim robot
                order = plan.get_execution_order() if hasattr(plan, "get_execution_order") else []
                from apyrobo.skills.executor import SkillExecutor
                executor = SkillExecutor(sim_robot)
                result = executor.execute_graph(plan)
                status_val = (
                    result.status.value
                    if hasattr(result.status, "value")
                    else str(result.status)
                )
                success = status_val == "completed"
                failures = [result.error] if result.error and not success else []
                return SimulationResult(
                    success=success,
                    steps_completed=result.steps_completed,
                    steps_total=result.steps_total,
                    failures=failures,
                    robot_final_position={},
                )
        except Exception as exc:
            logger.warning("Simulation validation failed with exception: %s", exc)
            return SimulationResult(
                success=False,
                steps_completed=0,
                steps_total=getattr(plan, "__len__", lambda: 0)(),
                failures=[str(exc)],
            )

    def deploy(self, plan: Any, agent: Any = None) -> bool:
        """Deploy *plan* to the real robot.

        Returns ``True`` on success, ``False`` if the deployment fails or if
        ``real_adapter_uri`` was not set.
        """
        if self._real_uri is None:
            logger.warning("deploy() called but real_adapter_uri is not set")
            return False

        try:
            from apyrobo.core.robot import Robot
            from apyrobo.core.schemas import TaskStatus
            real_robot = Robot.discover(self._real_uri)

            if agent is not None:
                if hasattr(plan, "get_execution_order"):
                    from apyrobo.skills.executor import SkillExecutor
                    executor = SkillExecutor(real_robot)
                    result = executor.execute_graph(plan)
                else:
                    result = agent.execute(str(plan), real_robot)
            else:
                from apyrobo.skills.executor import SkillExecutor
                executor = SkillExecutor(real_robot)
                result = executor.execute_graph(plan)

            status_val = (
                result.status.value
                if hasattr(result.status, "value")
                else str(result.status)
            )
            return status_val == "completed"
        except Exception as exc:
            logger.warning("Real robot deployment failed: %s", exc)
            return False

    def run(
        self,
        plan: Any,
        agent: Any = None,
        auto_deploy: bool = False,
        timeout_s: float = 60.0,
    ) -> tuple[SimulationResult, bool]:
        """Validate in sim, then optionally deploy to the real robot.

        Args:
            plan:        Skill graph or task to run.
            agent:       Optional agent for execution.
            auto_deploy: If True and simulation succeeded, deploy automatically.
            timeout_s:   Simulation timeout.

        Returns:
            Tuple of (SimulationResult, deployed: bool).
        """
        sim_result = self.validate(plan, agent=agent, timeout_s=timeout_s)
        deployed = False
        if auto_deploy and sim_result.success:
            deployed = self.deploy(plan, agent=agent)
        return sim_result, deployed


# ---------------------------------------------------------------------------
# MockSimToRealTransfer
# ---------------------------------------------------------------------------

class MockSimToRealTransfer:
    """Deterministic sim-to-real adapter for tests.

    Args:
        sim_result: The :class:`SimulationResult` returned by ``validate()``.
                    Defaults to a successful 1-step result.
        deploy_success: Whether ``deploy()`` returns True.
    """

    def __init__(
        self,
        sim_result: SimulationResult | None = None,
        deploy_success: bool = True,
    ) -> None:
        self._sim_result = sim_result or SimulationResult(
            success=True,
            steps_completed=1,
            steps_total=1,
        )
        self._deploy_success = deploy_success
        self.validate_calls: list[dict[str, Any]] = []
        self.deploy_calls: list[dict[str, Any]] = []

    def set_sim_result(self, result: SimulationResult) -> None:
        """Override the simulated result."""
        self._sim_result = result

    def validate(self, plan: Any, agent: Any = None, timeout_s: float = 60.0) -> SimulationResult:
        self.validate_calls.append({"plan": plan, "agent": agent, "timeout_s": timeout_s})
        return self._sim_result

    def deploy(self, plan: Any, agent: Any = None) -> bool:
        self.deploy_calls.append({"plan": plan, "agent": agent})
        return self._deploy_success

    def run(
        self,
        plan: Any,
        agent: Any = None,
        auto_deploy: bool = False,
        timeout_s: float = 60.0,
    ) -> tuple[SimulationResult, bool]:
        sim_result = self.validate(plan, agent=agent, timeout_s=timeout_s)
        deployed = False
        if auto_deploy and sim_result.success:
            deployed = self.deploy(plan, agent=agent)
        return sim_result, deployed
