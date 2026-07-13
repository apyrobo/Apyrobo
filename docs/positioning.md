# Positioning — who APYROBO is for, and why they'd pick it over building their own

Research snapshot, July 2026. This document grounds roadmap priorities in
who actually needs an AI-orchestration layer and what they'd otherwise use.

## The one-sentence claim

> ROS 2, VDA5050, MAVLink, and vendor SDKs are the *runtimes*. The APYROBO
> protocol is the orchestration API above all of them. Buying robots should
> be enough — the intelligence layer is open, standard, and free.

The Kubernetes analogy runs one level deeper than "the API contract won":
Kubernetes also won through **CRI**, the runtime interface that let
containerd and CRI-O plug in underneath. APYROBO's
[adapter contract](../spec/adapter-contract.md) is that interface; each base
below is a runtime.

## Personas

| Who | Situation | What they need | What they'd use instead |
|-----|-----------|----------------|------------------------|
| **Students / courses** | ROS 2 + TurtleBot3/4 is the standard undergrad stack; no standard "AI layer" module exists | Learn AI-robotics without ROS plumbing: `mock://` on a laptop → Gazebo lab → TB4 capstone | Hand-rolled rclpy scripts per course |
| **Research labs** | Riding the VLA wave (OpenVLA, π0, LeRobot); the deployment literature says neural policies **require runtime safety monitors** since they can't be certified | A safety envelope, execution monitoring, and recovery around learned policies — plus reproducible orchestration | Hand-rolled glue per paper |
| **Robot-buying SMBs** (warehouse, 3PL, agtech, inspection) | Can afford robots (AMR pilots $50–150K; RaaS makes them opex) but not a robotics software team; their AMRs increasingly speak **VDA5050** | An open master control with NL task input and WMS glue | Proprietary fleet managers or RobOps SaaS (InOrbit, Formant) — priced for enterprises |
| **Robotics product startups** | Building a robot product; orchestration/fleet/safety is undifferentiated heavy lifting | The middleware layer, without platform lock-in | Build in-house (2–4 engineer-years), or Viam (proprietary, cloud-first) |

## The landscape

- **[Viam](https://www.viam.com/)** — closest commercial analog: hardware
  abstraction, modular registry, fleet management; $30M raised. Their pitch
  minus AI-native planning, minus an open protocol. Differentiation:
  APYROBO's frozen spec + conformance suite + Apache-2.0 vs. platform
  lock-in.
- **[NASA JPL ROSA](https://github.com/nasa-jpl/rosa)** — closest OSS
  analog and demand validation: an LLM agent for ROS 1/2. It is an
  *operator assistant* (inspect/diagnose/operate); no skill graphs, safety
  enforcer, fleet, or wire protocol. APYROBO is the production-runtime
  superset.
- **[Open-RMF](https://www.robotics247.com/article/open_rmf_addresses_growing_interoperability_needs)**
  — multi-vendor fleet coordination (hospitals, airports). Not AI-native.
  Treat as a *bridge target* (APYROBO as the AI task layer above an RMF
  deployment), not a competitor.
- **InOrbit / Formant** — cloud "RobOps" SaaS: monitoring, analytics,
  orchestration; support VDA5050/MassRobotics/RMF. Ops dashboards, not
  planners; priced past the SMB persona.
- **DIY** — LangChain + rclpy scripts. Loses on safety, verification,
  recovery, fleet, and spec stability — *if* the wedges below exist. The
  replacement-cost argument: rebuilding APYROBO's runtime is 2–4
  engineer-years.

## Bases beyond ROS 2

| Base | World | Status in APYROBO |
|------|-------|-------------------|
| **ROS 2** | Research, education, most modern robots | Native (`ros2://`, flagship, CI-verified) |
| **VDA5050 (MQTT)** | Industrial AMR fleets — MiR, OTTO/Rockwell, Seegrid converging on it for 2026; no ROS required on the robot | **Wedge #1 — landed** (`vda5050://`, verified vs. simulated AGV; hardware = Arc 1 gate) |
| **MAVLink / MAVSDK** | PX4 drones (inspection, agtech) | Scaffold (`apyrobo-skills-drone-px4`) awaiting real wiring |
| **Vendor SDKs** | Spot (`bosdyn`), UR (`ur_rtde`), Franka | Scaffolds awaiting real wiring — each graduation is a market segment |
| **Zenoh** | ROS 2's own next transport (`rmw_zenoh`, Jazzy+); cloud↔edge | Watch; candidate wire-protocol binding for spec 1.1 |

## The two wedges (why these first)

1. **VDA5050 master control** — the AMR industry built the interop
   standard; what a compliant fleet lacks is an affordable master
   controller. This is a *distinct user population with no open option*,
   and it makes APYROBO's "protocol above any middleware" claim literally
   true (first non-ROS base).
2. **Policy-backed skills** — VLA deployment *requires* the thing APYROBO
   already is (safety envelope + runtime monitor + recovery around an
   uncertifiable policy). Rides the largest current wave in robotics;
   acquires the lab/course population that produces tomorrow's
   practitioners.

Both are adapter-shaped, both land inside the existing spec, and each is a
reason to adopt rather than a feature.

## Sources

[Viam](https://www.viam.com/product/platform-overview) ·
[Viam raise](https://www.therobotreport.com/viam-raises-30m-to-scale-robotics-development-platform/) ·
[VDA5050 guide](https://ottomotors.com/blog/interoperability-standard-vda5050/) ·
[MiR on VDA5050](https://mobile-industrial-robots.com/news-center/interoperability-the-vda5050-standard-and-mir-s-approach) ·
[Seegrid 2026 compliance](https://www.dcvelocity.com/material-handling/internal-movement/autonomous-mobile-robots-amrs/seegrid-amrs-to-meet-interoperability-standard-by-2026) ·
[ROSA](https://github.com/nasa-jpl/rosa) · [ROSA paper](https://arxiv.org/abs/2410.06472) ·
[VDA5050 vs MassRobotics vs Open-RMF](https://www.synaos.com/en/blog/vda-5050-massrobotics-open-rmf) ·
[VLA models 2026](https://www.roboticscenter.ai/tools/vla-models-comparison) ·
[π0 on LeRobot](https://huggingface.co/blog/pi0) ·
[Small-warehouse budgets](https://www.supplychain247.com/article/small-warehouse-automation-on-a-budget) ·
[Pilot costs](https://robotomated.com/learn/cost/warehouse-automation-budget-guide) ·
[rmw_zenoh](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds) ·
[InOrbit interop](https://www.inorbit.ai/interoperability)
