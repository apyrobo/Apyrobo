# RFC Process

The protocol spec ([`spec/`](../spec/)) is frozen at 1.0. Every change to it —
schema, behavior, or patch-level prose clarification — goes through a public
RFC. This is deliberate: the spec is a commons that other implementations
depend on, not one maintainer's notebook. The RFC process is what makes it
safe for a team we've never met to build against the spec.

## Lifecycle

```
Draft ──► Comment window ──► Final comment period ──► Accepted ──► ADR + spec PR
  │            (≥ 14 days)          (7 days)      └──► Rejected
  └──────────────────────────────────────────────────► Withdrawn (any time)
```

1. **Draft.** Open a GitHub issue with the
   [RFC template](../.github/ISSUE_TEMPLATE/rfc.yml) (`New issue → RFC: spec
   change proposal`). The template requires the motivation, the *exact*
   schema/prose diff, the revision level (patch/minor/major per
   [spec/README.md § Versioning](../spec/README.md#versioning)), the
   compatibility impact, and alternatives considered. Proposals without a
   concrete diff are sent back to draft — direction-setting discussions belong
   in a regular issue or Discussion first.

2. **Comment window.** The RFC stays open for public comment for **at least
   14 days** from the moment it has a complete diff. Maintainers participate
   as commenters like everyone else. Substantial revisions during the window
   restart the 14-day clock.

3. **Final comment period (FCP).** A maintainer announces the proposed
   disposition — *accept* or *reject* — with a one-paragraph rationale, and
   the RFC enters a **7-day** FCP. New blocking arguments during FCP cancel
   it and return the RFC to the comment window.

4. **Decision.** After an uneventful FCP the disposition stands. The issue is
   closed with the outcome label (`rfc-accepted` / `rfc-rejected`) and a
   closing comment summarizing the rationale, including the strongest
   objection and why it didn't carry.

5. **Record.** Every accepted RFC is recorded as an ADR in
   [`docs/adr/`](adr/) — the durable, in-repo record of what was decided and
   why, surviving issue-tracker migrations. The ADR lands in the same PR as
   the spec change, and the spec changelog line references both.

## Who decides

Disposition is proposed by any maintainer and stands unless another
maintainer objects during FCP; maintainer disagreement is resolved by
consensus among maintainers, in the open, on the issue. The bar for
*accepting* scales with the revision level: patch clarifications need one
maintainer and no objections; minor revisions need affirmative support from
implementers who would consume the change; major revisions additionally need
a negotiation/migration story before FCP can start.

## Enforcement

CI enforces the gate mechanically: a pull request that touches `spec/` fails
the [`spec-guard`](../.github/workflows/spec-guard.yml) check unless it
carries the `rfc-accepted` label or its description references the accepted
RFC issue (e.g. `RFC: #123`). The drift-guard tests
(`tests/test_spec_schemas.py`) then keep the reference implementation honest
against whatever the spec says.

## Scope

In scope: everything under `spec/` — the four prose documents and the JSON
Schemas. Out of scope (regular PRs, no RFC): the Python reference
implementation, the TS client, the conformance suite's *implementation*
(its *check catalog* follows the spec, so checks change when the spec does),
docs outside `spec/`, and this process document itself — though changes to
this document still deserve an issue and review like any other.
