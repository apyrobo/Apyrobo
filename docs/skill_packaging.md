# Creating and Publishing APYROBO Skill Packages

This guide shows you how to build a skill package that any APYROBO user can install with `pip`.
The reference implementation lives at `packages/apyrobo-skills-turtlebot4/`.

---

## 1. Package layout

```
apyrobo-skills-myrobot/
  pyproject.toml
  README.md
  apyrobo_skills_myrobot/
    __init__.py       ← exports all skill functions + register()
    navigation.py     ← @skill-decorated functions
    manipulation.py
  tests/
    test_skills.py
```

Use the module name `apyrobo_skills_<yourbot>` (underscores) for the importable package and
`apyrobo-skills-<yourbot>` (hyphens) for the PyPI distribution name.

---

## 2. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta:__legacy__"

[project]
name = "apyrobo-skills-myrobot"
version = "1.0.0"
description = "MyRobot skill pack for APYROBO"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.10"
dependencies = ["apyrobo>=1.0.0"]

# This entry-point tells APYROBO where to find your register() function.
[project.entry-points."apyrobo.skills"]
myrobot = "apyrobo_skills_myrobot:register"

[tool.setuptools.packages.find]
where = ["."]
include = ["apyrobo_skills_myrobot*"]
```

The `[project.entry-points."apyrobo.skills"]` section names your `register` function.
`apyrobo test-skill` loads it (see section 5). Nothing else in APYROBO loads it automatically
yet, so in their own code users import your package and call `register()` themselves (see
section 7).

---

## 3. Writing skills

Use the `@skill` decorator from `apyrobo`:

```python
# apyrobo_skills_myrobot/navigation.py
import time
from apyrobo import skill

@skill(
    description="Navigate to a named waypoint",
    capability="navigate",
)
def go_to_waypoint(waypoint_id: str = "wp_01") -> bool:
    print(f"  [go_to_waypoint] Navigating to {waypoint_id!r}")
    time.sleep(0.05)   # simulate motion — keep under 0.1 s for tests
    return True
```

Guidelines:
- **Return `bool`** — `True` for success, `False` for failure.
- **Print progress** — users running `apyrobo test-skill` see your output.
- **Keep stubs short** — the semantic layer describes *what* a skill does, not *how*.
  The adapter (ROS 2, Gazebo, mock) handles physical execution.
- **Clamp numeric inputs** — use `max(lo, min(hi, value))` to avoid nonsense values.
- **Default all params** — so skills can run with zero arguments.

---

## 4. `__init__.py` with `register()`

```python
# apyrobo_skills_myrobot/__init__.py
from apyrobo_skills_myrobot.navigation import go_to_waypoint

__all__ = ["go_to_waypoint", "register"]

def register() -> None:
    """Wire this package's @skill functions into APYROBO's skill registry."""
    from apyrobo.skills.library import SkillLibrary
    from apyrobo.skills.decorators import get_decorated_skills

    SkillLibrary.from_decorated()   # wires handlers into the global registry
    decorated = get_decorated_skills()
    skills = [sid for sid in decorated if sid in ("go_to_waypoint",)]
    print(f"[apyrobo-skills-myrobot] Registered {len(skills)} skill(s): " + ", ".join(skills))
```

---

## 5. Testing

```python
# tests/test_skills.py
from apyrobo_skills_myrobot.navigation import go_to_waypoint

def test_go_to_waypoint_returns_true():
    assert go_to_waypoint() is True

def test_go_to_waypoint_returns_bool():
    assert isinstance(go_to_waypoint(waypoint_id="wp_42"), bool)
```

Run:
```bash
pytest packages/apyrobo-skills-myrobot/tests/
```

`apyrobo test-skill` also runs skills from installed packages. When a skill ID is not a
built-in skill, it loads every installed `apyrobo.skills` entry point (calling the `register`
function each one names) and looks again. So once the package is installed
(`pip install -e packages/apyrobo-skills-myrobot`), this runs `go_to_waypoint` from it:
```bash
apyrobo test-skill go_to_waypoint --repeat 5
```

A `.py` file works too, installed or not: test-skill imports it and runs the skill named after
the file, or the file's only `@skill` when none is (a file with several skills and none named
after it is refused with their names):
```bash
# go_to_waypoint.py holds the @skill go_to_waypoint function from section 3
apyrobo test-skill go_to_waypoint.py --repeat 5
apyrobo test-skill go_to_waypoint.py --params '{"waypoint_id": "wp_42"}'
```

---

## 6. Publishing to PyPI

```bash
# Build
cd packages/apyrobo-skills-myrobot
pip install build
python -m build

# Upload (requires a PyPI account and API token)
pip install twine
twine upload dist/*
```

Then publish to the APYROBO hosted registry so it shows up in `apyrobo skill search`:

```bash
apyrobo skill publish go_to_waypoint \
  --name apyrobo-skills-myrobot \
  --version 1.0.0 \
  --download-url https://pypi.org/packages/... \
  --token $APYROBO_REGISTRY_TOKEN
```

---

## 7. Installation and discovery

Users install your package:
```bash
pip install apyrobo-skills-myrobot
```

And load it by importing it and calling `register()`:
```python
import apyrobo_skills_myrobot

apyrobo_skills_myrobot.register()   # wires its @skill functions into the skill registry
```

Or by name:
```bash
apyrobo skill search myrobot
```

---

## Reference implementation

See `packages/apyrobo-skills-turtlebot4/` for a complete example with six skills across
three modules (navigation, inspection, social).
