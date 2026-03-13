"""
APYROBO Inference — edge/cloud AI routing and management.

The InferenceRouter sits between the Agent and LLM providers,
dynamically routing requests based on urgency, latency, and
provider health. Safety-critical operations (motor commands,
sensor reading, safety enforcement) never go through inference —
they always run locally.

Key components:
    - router: InferenceRouter with multi-tier fallback
    - Urgency levels: HIGH (edge), NORMAL (cloud preferred), LOW (batch OK)
    - Health tracking: per-provider latency, error rate, availability
"""
