# Decision log

Append-only. Newest at the bottom. One entry per run.

Format:

```
## YYYY-MM-DD — <research|build> — <short title>
- Decided: …
- Why: …
- Evidence: links / files / CI runs
- PR: #…
```

## Needs human

Decisions agents have prepared but must not make themselves. Remove an item
once a human has decided (and note the outcome in the log).

- Publish the spec v1.0 launch post (`docs/announcing_spec_1_0.md`) — where and when.

---

## 2026-10-29 — setup — Agent system created
- Decided: run a daily research agent and a weekday build agent, both PR-only, with this folder as shared memory.
- Why: owner wants work to continue without daily prompting; previous plan existed only in chat and was lost.
- Evidence: `ROADMAP.md` (v8 Phase 3 + Arcs 1–4), last commit on main 2026-07-19.
- PR: (this PR)
