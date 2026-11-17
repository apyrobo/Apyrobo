"""Voice-commanded robot — listen for spoken instructions and execute them.

Each heard command goes through ``apyrobo.voice.voice_loop``: the same
safety policy and plan checks as ``apyrobo voice --listen`` and
``apyrobo mcp``, so a command the rule planner cannot read (a place name
like "the kitchen") is refused with the reason instead of driving somewhere
made up. The rule planner reads coordinates in metres: say
"go to (2, 1)". Saying "stop" ends the session. The script exits 1 if any
command was refused or failed.

The mock adapter plays the demo commands below. Whisper needs
``pip install openai-whisper`` and a microphone.

Usage:
    python examples/workflows/voice_commanded_robot.py
    python examples/workflows/voice_commanded_robot.py --adapter whisper --robot ros2://turtlebot4
"""
import argparse
import sys

from apyrobo import Agent, Robot
from apyrobo.core.schemas import TaskStatus
from apyrobo.voice import MockVoiceAdapter, WhisperVoiceAdapter, voice_loop

DEMO_UTTERANCES = [
    "go to (2, 1)",
    "rotate 90 degrees",
    "go to (0, -1.5)",
    "stop",
]


def run_voice_loop(robot_uri: str, adapter_name: str, model: str, max_commands: int) -> bool:
    """Run up to ``max_commands`` commands; False if any was refused or failed."""
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    if adapter_name == "whisper":
        adapter = WhisperVoiceAdapter(model=model)
        print(f"[voice] Whisper STT (model={model})")
    else:
        adapter = MockVoiceAdapter(listen_responses=DEMO_UTTERANCES)
        print(f"[voice] mock adapter  ({len(DEMO_UTTERANCES)} demo commands)")

    print(f"[voice] robot={robot_uri}  listening … (say 'stop' to end)\n")

    def on_listen(text: str) -> None:
        print(f"  heard: {text!r}")

    def on_result(result) -> None:
        if result.status == TaskStatus.COMPLETED:
            print(f"  ✓ done ({result.steps_completed}/{result.steps_total} steps)\n")
        else:
            print(f"  ✗ {result.status.value}: {result.error or 'no reason given'}\n")

    try:
        turns = voice_loop(
            agent=agent,
            robot=robot,
            adapter=adapter,
            max_turns=max_commands,
            on_listen=on_listen,
            on_result=on_result,
            robot_uri=robot_uri,
        )
    except ImportError as exc:
        print(f"[voice] {adapter_name} is not installed: {exc}")
        return False

    failed = [t for t in turns if t["result"].status != TaskStatus.COMPLETED]
    print(f"[voice] session ended: {len(turns) - len(failed)} command(s) done, "
          f"{len(failed)} refused or failed")
    return not failed


def main() -> None:
    p = argparse.ArgumentParser(description="Voice-commanded robot workflow")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--adapter", default="mock", choices=["mock", "whisper"],
                   help="Voice adapter (mock or whisper)")
    p.add_argument("--model", default="base", help="Whisper model size (base/small/medium)")
    p.add_argument("--max-commands", type=int, default=5)
    args = p.parse_args()
    if not run_voice_loop(args.robot, args.adapter, args.model, max(1, args.max_commands)):
        sys.exit(1)


if __name__ == "__main__":
    main()
