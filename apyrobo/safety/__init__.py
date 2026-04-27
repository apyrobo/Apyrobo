"""
APYROBO Safety — Safety Policy Enforcement.

Hard constraints enforced at the framework level, independent of
what any AI agent requests.  No agent can bypass safety.

Key components (Phase 3):
    - policy: SafetyPolicy schema
    - enforcer: SafetyEnforcer that wraps all robot commands
    - envelope: Speed caps, collision zones, human proximity limits
    - verification: Formal safety property verification and certification
"""

from apyrobo.safety.verification import (  # noqa: F401
    SafetyVerifier,
    SafetyProperty,
    VerificationResult,
    CertificationReport,
    generate_certification_report,
    BUILTIN_PROPERTIES,
)
