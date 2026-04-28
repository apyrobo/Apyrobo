"""Tests for @skill decorator, SkillLibrary.from_decorated(), and Skill.simple()."""

from __future__ import annotations

import pytest

from apyrobo.skills.decorators import skill, get_decorated_skills, clear_decorated_skills
from apyrobo.skills.skill import Skill
from apyrobo.skills.library import SkillLibrary
from apyrobo.core.schemas import CapabilityType


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure each test starts with a clean decorator registry."""
    clear_decorated_skills()
    yield
    clear_decorated_skills()


# ---------------------------------------------------------------------------
# Bare @skill (no parentheses)
# ---------------------------------------------------------------------------

class TestBareDecorator:
    def test_function_is_still_callable(self):
        @skill
        def greet(name: str) -> str:
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_skill_attribute_set(self):
        @skill
        def greet(name: str) -> str:
            return f"hello {name}"

        assert hasattr(greet, "__skill__")
        assert isinstance(greet.__skill__, Skill)

    def test_skill_id_attribute_set(self):
        @skill
        def greet(name: str) -> str:
            return "hi"

        assert greet.__skill_id__ == "greet"

    def test_skill_id_matches_function_name(self):
        @skill
        def do_the_thing() -> None:
            pass

        s = do_the_thing.__skill__
        assert s.skill_id == "do_the_thing"

    def test_name_auto_derived(self):
        @skill
        def inspect_shelf() -> None:
            pass

        assert inspect_shelf.__skill__.name == "Inspect Shelf"

    def test_registered_in_global_dict(self):
        @skill
        def my_action() -> None:
            pass

        assert "my_action" in get_decorated_skills()


# ---------------------------------------------------------------------------
# @skill(...)  with keyword arguments
# ---------------------------------------------------------------------------

class TestDecoratorWithArgs:
    def test_description_set(self):
        @skill(description="Pick up a cup")
        def pick_cup() -> None:
            pass

        assert pick_cup.__skill__.description == "Pick up a cup"

    def test_capability_set(self):
        @skill(capability="pick")
        def pick_cup() -> None:
            pass

        assert pick_cup.__skill__.required_capability == CapabilityType.PICK

    def test_timeout_set(self):
        @skill(timeout=30.0)
        def quick_action() -> None:
            pass

        assert quick_action.__skill__.timeout_seconds == 30.0

    def test_retries_set(self):
        @skill(retries=3)
        def flaky_action() -> None:
            pass

        assert flaky_action.__skill__.retry_count == 3

    def test_skill_id_override(self):
        @skill(skill_id="custom_pick")
        def pick_cup() -> None:
            pass

        assert pick_cup.__skill_id__ == "custom_pick"
        assert pick_cup.__skill__.skill_id == "custom_pick"
        assert "custom_pick" in get_decorated_skills()

    def test_name_override(self):
        @skill(name="My Custom Name")
        def some_action() -> None:
            pass

        assert some_action.__skill__.name == "My Custom Name"

    def test_function_still_callable_with_args(self):
        @skill(description="Navigate somewhere")
        def navigate(x: float, y: float) -> tuple[float, float]:
            return (x, y)

        assert navigate(1.0, 2.0) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Parameter extraction from type annotations
# ---------------------------------------------------------------------------

class TestParameterExtraction:
    def test_str_annotation_gives_empty_string_default(self):
        @skill
        def pick(object_id: str) -> None:
            pass

        params = pick.__skill__.parameters
        assert "object_id" in params
        assert params["object_id"] == ""

    def test_float_annotation_gives_zero_default(self):
        @skill
        def move(speed: float) -> None:
            pass

        assert move.__skill__.parameters["speed"] == 0.0

    def test_int_annotation_gives_zero_default(self):
        @skill
        def repeat(count: int) -> None:
            pass

        assert repeat.__skill__.parameters["count"] == 0

    def test_explicit_default_takes_precedence_over_annotation(self):
        @skill
        def move(speed: float = 1.5) -> None:
            pass

        assert move.__skill__.parameters["speed"] == 1.5

    def test_multiple_params(self):
        @skill
        def navigate(x: float, y: float, label: str = "home") -> None:
            pass

        params = navigate.__skill__.parameters
        assert params["x"] == 0.0
        assert params["y"] == 0.0
        assert params["label"] == "home"

    def test_robot_param_excluded(self):
        @skill
        def do_thing(robot, speed: float) -> None:
            pass

        assert "robot" not in do_thing.__skill__.parameters
        assert "speed" in do_thing.__skill__.parameters


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

class TestRegistryHelpers:
    def test_get_decorated_skills_returns_snapshot(self):
        @skill
        def alpha() -> None:
            pass

        @skill
        def beta() -> None:
            pass

        d = get_decorated_skills()
        assert "alpha" in d
        assert "beta" in d

    def test_snapshot_is_copy(self):
        @skill
        def alpha() -> None:
            pass

        d = get_decorated_skills()
        d["injected"] = None  # mutating the snapshot
        assert "injected" not in get_decorated_skills()

    def test_clear_decorated_skills(self):
        @skill
        def alpha() -> None:
            pass

        clear_decorated_skills()
        assert get_decorated_skills() == {}

    def test_get_decorated_skills_value_is_tuple(self):
        @skill
        def alpha() -> None:
            pass

        skill_def, fn = get_decorated_skills()["alpha"]
        assert isinstance(skill_def, Skill)
        assert callable(fn)


# ---------------------------------------------------------------------------
# SkillLibrary.from_decorated()
# ---------------------------------------------------------------------------

class TestSkillLibraryFromDecorated:
    def test_from_decorated_includes_registered_skills(self):
        @skill(description="Test skill")
        def test_action() -> None:
            pass

        lib = SkillLibrary.from_decorated()
        assert "test_action" in lib

    def test_from_decorated_skill_accessible_via_get(self):
        @skill
        def do_thing(x: float) -> None:
            pass

        lib = SkillLibrary.from_decorated()
        s = lib.get("do_thing")
        assert s is not None
        assert s.skill_id == "do_thing"

    def test_from_decorated_empty_when_none_registered(self):
        lib = SkillLibrary.from_decorated()
        custom = lib.custom_skills()
        assert custom == {}


# ---------------------------------------------------------------------------
# Skill.simple()
# ---------------------------------------------------------------------------

class TestSkillSimple:
    def test_creates_skill_with_correct_id(self):
        s = Skill.simple("pick_cup", "Pick up a cup")
        assert s.skill_id == "pick_cup"

    def test_auto_derives_name(self):
        s = Skill.simple("pick_cup", "Pick up a cup")
        assert s.name == "Pick Cup"

    def test_description_set(self):
        s = Skill.simple("go_home", "Return to dock")
        assert s.description == "Return to dock"

    def test_capability_parameter(self):
        s = Skill.simple("pick_cup", capability="pick")
        assert s.required_capability == CapabilityType.PICK

    def test_default_capability_is_custom(self):
        s = Skill.simple("my_action")
        assert s.required_capability == CapabilityType.CUSTOM

    def test_params_from_kwargs(self):
        s = Skill.simple("pick_cup", "Pick", object_id="", speed=0.5)
        assert s.parameters["object_id"] == ""
        assert s.parameters["speed"] == 0.5

    def test_timeout_and_retries(self):
        s = Skill.simple("retry_pick", timeout=30.0, retries=2)
        assert s.timeout_seconds == 30.0
        assert s.retry_count == 2

    def test_returns_skill_instance(self):
        s = Skill.simple("my_action")
        assert isinstance(s, Skill)
