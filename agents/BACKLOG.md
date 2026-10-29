# Backlog

Ranked top to bottom. Status: `ready` · `in-pr` (link) · `blocked` (reason) · `needs-human` · `done`.
The research agent re-ranks; the build agent takes the top `ready` item.

| # | Item | Roadmap link | Status | Notes |
|---|------|--------------|--------|-------|
| 1 | Health check: run full test suite + lint on `main`, fix anything red | Priority 1 | ready | First build run should do this to establish a baseline |
| 2 | Worked real-VLA-checkpoint example for policy-backed skills (LeRobot, optional dep, runnable without GPU via a tiny checkpoint) | Wedge: policy skills | ready | Keep core torch-free |
| 3 | Registry seeding: identify 3 candidate community skills and write mentored good-first-issue texts for them | Phase 3: registry | ready | Issue text goes in research brief; filing issues is fine |
| 4 | gz-sim / ros_gz port: design note + spike of the `gazebo-nav` compose profile on Gazebo Harmonic | Arc 3 | ready | Spike on a branch; don't break the existing CI job |
| 5 | Decide `gazebo_native://` and `isaac://`: write an ADR proposal (bridge vs remove) | Arc 3 | ready | Decision itself is `needs-human` once the ADR is drafted |
| 6 | VDA 5050: extend simulated-AGV tests to cover error/fault states and reconnects | Wedge: VDA 5050 | ready | |
| 7 | Spec v1.0 announcement: final review of `docs/announcing_spec_1_0.md` against current code | Phase 3: announcement | needs-human | Agents may prepare edits; posting is human |
