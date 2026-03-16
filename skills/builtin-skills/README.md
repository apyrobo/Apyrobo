# builtin-skills

Core APYROBO skill handlers for navigation, manipulation, and status reporting.

## Skills

| Skill ID | Description | Capability |
|----------|-------------|------------|
| `navigate_to` | Move the robot to (x, y) | navigate |
| `stop` | Halt all motion | navigate |
| `rotate` | Rotate in place by angle | rotate |
| `pick_object` | Close gripper to pick up | pick |
| `place_object` | Open gripper to release | place |
| `report_status` | Report capabilities and position | custom |
| `report_battery_status` | Report battery level | custom |

## Installation

```bash
apyrobo pkg install skills/builtin-skills
```

## License

Apache-2.0
