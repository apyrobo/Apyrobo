"""Voice-commanded robot — listen for spoken instructions and execute them.

Uses the voice adapter pipeline: STT → plan → execute. Works in mock mode
(reads tasks from a list) or with a real Whisper backend.

Usage:
    python examples/workflows/voice_commanded_robot.py
    python examples/workflows/voice_commanded_robot.py --adapter whisper --robot ros2://turtlebot4
"""
import argparse

from apyrobo import Robot, Agent
from apyrobo.voice import MockVoiceAdapter, WhisperVoiceAdapter

DEMO_UTTERANCES = [
    "navigate to the kitchen",
    "go to charging dock",
    "move to conference room B",
    "stop",
    "return to base",
]


def run_voice_loop(robot_uri: str, adapter_name: str, model: str, max_commands: int) -> None:
    robot = Robot.discover(robot_uri)
    agent = Agent(provider="rule")

    if adapter_name == "whisper":
        adapter = WhisperVoiceAdapter(model=model)
        print(f"[voice] Whisper STT (model={model})")
    else:
        adapter = MockVoiceAdapter(listen_responses=DEMO_UTTERANCES[:max_commands])
        print(f"[voice] mock adapter  ({len(DEMO_UTTERANCES[:max_commands])} demo commands)")

    print(f"[voice] robot={robot_uri}  listening …\n")

    count = 0
    while count < max_commands:
        try:
            utterance = adapter.listen()
        except StopIteration:
            print("  [end of demo commands]")
            break

        if not utterance:
            continue

        print(f"  heard: {utterance!r}")

        if utterance.lower().strip() in ("stop", "halt", "exit", "quit"):
            print("  → stop command received")
            robot.stop()
            break

        graph = agent.plan(utterance, robot)
        order = graph.get_execution_order()
        print(f"  → {len(order)} skill(s): {[s.name for s in order]}")
        agent.execute(task=utterance, robot=robot)
        count += 1
        print()

    print(f"[voice] session complete — {count} command(s) executed")


def main() -> None:
    p = argparse.ArgumentParser(description="Voice-commanded robot workflow")
    p.add_argument("--robot", default="mock://turtlebot4")
    p.add_argument("--adapter", default="mock", choices=["mock", "whisper"],
                   help="Voice adapter (mock or whisper)")
    p.add_argument("--model", default="base", help="Whisper model size (base/small/medium)")
    p.add_argument("--max-commands", type=int, default=5)
    args = p.parse_args()
    run_voice_loop(args.robot, args.adapter, args.model, args.max_commands)


if __name__ == "__main__":
    main()
