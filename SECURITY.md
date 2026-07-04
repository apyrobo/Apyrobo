# Security Policy

APYROBO orchestrates physical robots. A security vulnerability here can have
physical consequences, so reports are treated with priority — and the safety
paths (`stop()`, the safety enforcer, watchdogs, failover) are always in scope.

## Reporting a vulnerability

**Do not open a public issue for security reports.**

Report privately via
[GitHub Security Advisories](https://github.com/apyrobo/Apyrobo/security/advisories/new)
("Report a vulnerability"). Include:

- Affected version(s) and module(s)
- Reproduction steps or a proof of concept
- Impact assessment — especially whether robot motion, safety limits, or
  authentication can be influenced

You will receive an acknowledgement within **72 hours** and a triage decision
within **14 days**. We ask for coordinated disclosure: give us 90 days (or a
mutually agreed window) before publishing details. Credit is given in the
advisory and CHANGELOG unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---------|-----------|
| 4.x (current stable) | ✅ security fixes |
| < 4.0 | ❌ upgrade — see MIGRATION.md |

LTS designations and EOL dates are tracked by `apyrobo.lts` and published in
release notes.

## Scope notes for researchers

- **In scope:** the safety layer, RBAC/auth (`apyrobo.auth`), the audit trail's
  integrity guarantees, the REST API gateway, the skill registry
  (client and server), plugin loading, and all adapters.
- **Known non-guarantees (not vulnerabilities):** the raw orchestration wire
  protocol has no built-in authentication — this is documented in
  [spec/wire-protocol.md](spec/wire-protocol.md) §5 and deployments must front
  it with an authenticated transport. Reports demonstrating escalation
  *despite* the documented deployment guidance are very much in scope.
- The natural-language planning path treats LLM output as untrusted: plans are
  validated against the capability model and safety policies before execution.
  Bypasses of that validation are high-severity findings.

## Supply chain

- Releases are built and published from CI via PyPI Trusted Publishing (OIDC);
  no long-lived tokens exist.
- Every release ships a CycloneDX SBOM as a release asset.
- Runtime dependencies are intentionally few (see `pyproject.toml`); the v8
  roadmap's `apyrobo-core` split reduces them further.
