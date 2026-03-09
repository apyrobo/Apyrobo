# Contributing to APYROBO

Thanks for your interest in APYROBO! We're currently building privately toward our MVP demo. Once the repository goes public, we'll actively welcome contributions.

## When We Open Up

After the v0.1.0-mvp release, we'll accept contributions in these areas:

- **New capability adapters** — support for additional robot platforms
- **Skill definitions** — reusable skill JSON files for common tasks
- **Bug reports and fixes**
- **Documentation improvements**
- **Test coverage improvements**

## Development Setup

```bash
# Clone the repo
git clone https://github.com/apyrobo/apyrobo.git
cd apyrobo

# Option A: Docker (recommended)
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec apyrobo bash

# Option B: Local (Python only, no ROS 2 / Gazebo)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
ruff format .

# Type check
mypy apyrobo/
```

## Code Style

- **Python 3.10+** — use modern syntax (type unions with `|`, etc.)
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Pydantic v2** for all data schemas
- **Docstrings** on all public classes and functions
- **Tests** for all new functionality — target 80% coverage

## Commit Messages

Use clear, descriptive commit messages:

```
feat(core): add TurtleBot4 capability adapter
fix(safety): enforce speed cap on all move commands
test(skills): add skill chaining integration test
docs: update architecture diagram
```

## Questions?

Join our [Discord](#) or open a GitHub Discussion.
