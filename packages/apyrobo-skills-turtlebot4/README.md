# apyrobo-skills-turtlebot4

TurtleBot 4 skill pack for [APYROBO](https://github.com/apyrobo/Apyrobo).  
Provides navigation, inspection, and social behaviours as `@skill`-decorated functions.

## Install

```bash
pip install apyrobo-skills-turtlebot4
```

## Quick start

```python
from apyrobo import Robot, Agent
from apyrobo.skills.library import SkillLibrary

import apyrobo_skills_turtlebot4  # side-effect: registers all @skill functions

robot = Robot.discover("mock://turtlebot4")
agent = Agent(provider="rule", library=SkillLibrary.from_decorated())
result = agent.execute("patrol the area", robot=robot)
print(result.status)
```

## Skills

| Skill | Description | Key params |
|---|---|---|
| `patrol_area` | Navigate through waypoints N times | `waypoints`, `loops` |
| `dock` | Navigate to and dock at a charging station | `dock_station_id` |
| `undock` | Leave the charging dock | — |
| `inspect_room` | Systematic 360° camera scan of a room | `room_id`, `camera_height` |
| `check_surroundings` | 360° LIDAR + RGB-D sweep at current position | `radius` |
| `follow_person` | Track and follow a detected person | `person_id`, `duration_s` |

## Using with the plugin system

When installed, this package registers itself automatically via the `apyrobo.skills`
entry-point.  Call `SkillLibrary.from_plugins()` to pick up all installed skill packs:

```python
from apyrobo.skills.library import SkillLibrary

lib = SkillLibrary.from_plugins()   # discovers all apyrobo.skills entry-points
```

Or register explicitly:

```python
from apyrobo_skills_turtlebot4 import register
register()
```

## Testing skills

```bash
apyrobo test-skill patrol_area --repeat 3
apyrobo test-skill inspect_room --params '{"room_id": "lab_A", "camera_height": 1.0}'
```
