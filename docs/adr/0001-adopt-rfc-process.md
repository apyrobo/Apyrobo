# ADR-0001: Adopt a public RFC process for spec changes

- **Status:** Accepted
- **Date:** 2026-07-18
- **RFC:** n/a — bootstrap decision; the process cannot gate its own adoption
- **Spec revision:** n/a (process, not spec content)

## Context

Spec 1.0 froze in July 2026 with a one-paragraph note in
[spec/README.md](../../spec/README.md) saying changes "go through an RFC" —
but there was no template, no defined comment window, no decision rule, no
durable record of outcomes, and no enforcement. A frozen spec whose change
process is undefined is frozen only until the first contentious change.
Serious adopters — the labs, courses, and startups Arc 2 targets — need
evidence that the protocol is a commons with predictable governance, not one
maintainer's whim.

## Decision

Spec changes go through a public RFC: a structured issue template requiring
an exact diff and compatibility analysis, a ≥14-day public comment window, a
7-day final comment period with an announced disposition, and an ADR in
`docs/adr/` recording every accepted RFC. CI (`spec-guard`) fails any PR
touching `spec/` that doesn't reference an accepted RFC. The full process is
[docs/rfc_process.md](../rfc_process.md).

## Consequences

Easier: adopting or forking against the spec with confidence; onboarding
outside participants into governance (Arc 2's gate); pointing to a paper
trail when a decision is questioned. Harder: even a typo fix in `spec/` now
takes a 14-day window — accepted deliberately, because a spec that changes
casually isn't frozen. The strongest objection is process weight for a
project with few outside users today; it didn't carry because the process
existing *before* outside users arrive is precisely what makes arriving
safe.
