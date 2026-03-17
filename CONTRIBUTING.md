# Contributing to APYROBO

Thanks for your interest in APYROBO! We're currently building privately toward our MVP demo. Once the repository goes public, we'll actively welcome contributions.

## Where to Start

Check the [ROADMAP.md](ROADMAP.md) for the full list of contribution opportunities. Items are labeled:

- ![good first issue](https://img.shields.io/badge/-good%20first%20issue-7057ff) — Well-scoped, mentored, great for newcomers
- ![help wanted](https://img.shields.io/badge/-help%20wanted-008672) — Community contributions welcome, may require domain expertise

### Good First Issues

| Task | Milestone | Difficulty |
|------|-----------|------------|
| Increase test coverage to 90% (voice, handlers, edge cases) | v0.2.0 | Easy |
| Create a new skill package (e.g. patrol, inspection) | v0.2.0 | Easy |
| Add Gazebo adapter improvements (spawn/delete/reset) | v0.2.0 | Easy-Medium |
| Write v0.x → v1.0 migration guide | v1.0.0 | Easy |
| Add Kubernetes deployment template | v0.4.0 | Easy-Medium |

### Help Wanted

| Task | Milestone | Difficulty |
|------|-----------|------------|
| Voice adapter — Whisper STT + Piper TTS | v0.2.0 | Medium |
| VLM integration — camera-informed planning | v0.3.0 | Hard |
| MuJoCo simulation adapter | v0.3.0 | Medium |
| Formal safety verification proofs | v0.3.0 | Hard |
| Nav2 / MoveIt ROS 2 adapters | v0.2.0 | Medium-Hard |

### Other Contributions

We also welcome:

- **Bug reports and fixes**
- **Documentation improvements**
- **New capability adapters** for additional robot platforms
- **Skill definitions** — reusable skill JSON files for common tasks

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
