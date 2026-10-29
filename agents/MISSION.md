# Mission

Make APYROBO the standard for AI-native robotics orchestration — the thing
other people build *on*. Source of truth for direction is `ROADMAP.md`;
this file says how agents should prioritise inside it.

## Priorities (in order)

1. **Keep `main` healthy.** Red CI, broken install, failing nightly, or a doc
   claim the code doesn't back → fix first.
2. **Finish v8 Phase 3:** seeding the registry with real, installable
   packages; supporting the spec v1.0 announcement (the posting itself is a
   human decision — agents prepare, humans publish).
3. **Wedge follow-ups:** a worked real-VLA-checkpoint example for
   policy-backed skills; VDA 5050 hardening ahead of physical validation.
4. **Arc 3 — no stand-ins:** port Gazebo Classic → `gz-sim`/`ros_gz`; decide
   `gazebo_native://` and `isaac://` (real bridge or removal).
5. **Arc 2 groundwork:** good-first-issues, contributor docs, adapter
   authoring polish — things that make outside teams succeed.
6. **Arc 1 / Arc 4:** prepare what can be prepared without hardware (HIL
   runner docs, distributed executor design notes).

## What "good" looks like for a run

- Research: 3–7 concrete, sourced findings; backlog re-ranked with a one-line
  reason per move; nothing invented.
- Build: one small PR that a reviewer can approve in under 15 minutes, tests
  green, CHANGELOG `[Unreleased]` updated when user-facing.

## Owner

@apyrobo maintainers review all agent PRs. Edit this file to steer the agents.
