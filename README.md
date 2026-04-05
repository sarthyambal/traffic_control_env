---
title: Traffic Control Env — Autonomous Traffic Controller
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🚦 Traffic Control Env — OpenEnv Environment

An **Autonomous Traffic Control** simulator where LLM agents act as **City Traffic Controllers**. Agents must use a set of specialized tools to resolve traffic gridlocks, respond to accidents, and route emergency vehicles through a simulated city grid — all within a limited number of steps.

Built for the **Meta PyTorch OpenEnv Hackathon**.

---

## Overview

The environment presents the agent with a city grid of intersections, each with:
- A **traffic light phase** (`NS_GREEN` or `EW_GREEN`)
- A **vehicle queue** in each direction (N, S, E, W)
- Optional **accidents** or **emergency vehicles**

The agent must call the right tools in the right order to clear congestion, dispatch emergency services, and summarize the resolution — all while avoiding duplicate or incorrect tool calls.

---

## Quick Start

### Connect via Python

```python
from traffic_control_env import TrafficControlAction, TrafficControlEnv

# Connect to a running server
env = TrafficControlEnv(base_url="http://localhost:8000")

# Start an episode (pick a scenario)
result = await env.reset(scenario_id="easy_rush_hour")
obs = result.observation

print(f"Scenario: {obs.customer_query}")
print(f"Available tools: {[t['name'] for t in obs.available_tools]}")

# Execute a tool action
action = TrafficControlAction(
    tool_name="get_intersection_status",
    tool_args={"intersection_id": "INT-1"}
)
result = await env.step(action)
print(f"Tool result: {result.observation.tool_result}")
print(f"Reward: {result.reward}")
```

### Connect to a Running Server Directly

```python
env = TrafficControlEnv(base_url="http://localhost:8000")
```

---

## Scenarios

The environment includes 6 scenarios of increasing difficulty:

| Scenario ID | Difficulty | Description |
|---|---|---|
| `easy_rush_hour` | Easy | Heavy N-S traffic at a single intersection. Set the light correctly. |
| `medium_accident_response` | Medium | Accident at INT-2. Dispatch emergency and clear traffic. |
| `hard_gridlock_unbraiding` | Hard | 3-intersection deadlock. Sequential light changes required. |
| `hard_emergency_routing` | Hard | Ambulance needs a green corridor through INT-1 and INT-2. |
| `expert_cascade_emergency` | Expert | 3-intersection chain blocked by accident — restore flow to entire corridor. |
| `expert_ring_deadlock` | Expert | 4-intersection ring deadlock — find the correct unlock order. |

---

## Available Tools

Agents interact with the environment exclusively through these tools:

### `get_network_status`
Returns a high-level overview of all intersections — identifies accidents, emergencies, and gridlock severity.
```json
{ "tool_name": "get_network_status", "tool_args": {} }
```

### `get_intersection_status`
Returns the detailed queue lengths and light phase of a specific intersection.
```json
{ "tool_name": "get_intersection_status", "tool_args": { "intersection_id": "INT-1" } }
```

### `set_traffic_light`
Changes the traffic light phase of an intersection to allow traffic to flow.
```json
{
  "tool_name": "set_traffic_light",
  "tool_args": { "intersection_id": "INT-1", "phase": "NS_GREEN" }
}
```
`phase` must be `"NS_GREEN"` (North-South green) or `"EW_GREEN"` (East-West green).

### `dispatch_emergency_vehicle`
Dispatches emergency services to clear an accident at an intersection.
```json
{ "tool_name": "dispatch_emergency_vehicle", "tool_args": { "intersection_id": "INT-2" } }
```

---

## Action & Observation Format

### Action (`TrafficControlAction`)

```python
TrafficControlAction(
    tool_name="set_traffic_light",           # Name of the tool to call (or None)
    tool_args={"intersection_id": "INT-1",   # Arguments for the tool
               "phase": "NS_GREEN"},
    message="Cleared N-S queue at INT-1."    # Optional resolution summary
)
```

The `message` field is used to signal the end of an episode when it contains **resolution keywords** (e.g. "cleared", "dispatched", "resolved"). This is how the agent declares it has completed the scenario.

### Observation (`TrafficControlObservation`)

| Field | Type | Description |
|---|---|---|
| `customer_query` | `str` | The scenario description |
| `tool_result` | `dict` | Result from the last tool call |
| `feedback` | `str` | System feedback on the last action (reward info, penalties) |
| `available_tools` | `list[dict]` | Full tool schemas available this step |
| `scenario_id` | `str` | Active scenario identifier |
| `difficulty` | `str` | `"easy"`, `"medium"`, or `"hard"` |
| `steps_taken` | `int` | Steps used so far |
| `max_steps` | `int` | Maximum steps per episode (20) |
| `reward` | `float` | Reward for the last step |
| `done` | `bool` | Whether the episode has ended |

---

## Reward Structure

Rewards are designed to incentivize correct tool sequencing, efficient resolution, and accurate diagnosis:

| Action | Reward |
|---|---|
| `get_network_status` | `+0.1` to `+0.2` (per scenario) |
| `get_intersection_status` | `+0.1` to `+0.2` |
| `set_traffic_light` (correct direction) | `+0.1` to `+0.5` |
| `dispatch_emergency_vehicle` | `+0.4` (medium scenario) |
| Resolution message with correct keywords | `+0.0` to `+0.2` |
| Duplicate tool call (same args) | `-0.03` penalty |
| Wrong tool for scenario | `-0.05` penalty |
| Total score capped at | `1.00` |

---

## Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# Push to a specific repository
openenv push --repo-id my-org/traffic-control-env

# Push as private
openenv push --private
```

After deployment, your space will be available with:
- **Web Interface** at `/web` — Interactive playground to test the environment
- **API Documentation** at `/docs` — Full Swagger/OpenAPI interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent low-latency sessions

---

## Running Locally

### With Docker (recommended)

```bash
docker build -t traffic_control_env:latest .
docker run -p 8000:8000 traffic_control_env:latest
```

Then open: [http://localhost:8000/web/](http://localhost:8000/web/)

### Without Docker

```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running the Oracle / Benchmark

A hardcoded oracle suite validates that the environment produces the expected scores:

```bash
python test_benchmark.py
```

Expected output:
```
=== Running OpenEnv Oracle Test Suite ===
[EASY_RUSH_HOUR] Result: Score 0.90+ / 1.0
[MEDIUM_ACCIDENT_RESPONSE] Result: Score 1.00 / 1.0
[HARD_GRIDLOCK_UNBRAIDING] Result: Score 1.00 / 1.0
[HARD_EMERGENCY_ROUTING] Result: Score 1.00 / 1.0

=== Final Avg Score: 0.97 ===
```

---

## Running the LLM Agent

```bash
# Set your API key
export HF_TOKEN=your_hf_token

# Run inference on a scenario
TRAFFIC_ENV_TASK=easy_rush_hour python inference.py
```

---

## Project Structure

```
traffic_control_env/
├── __init__.py                          # Module exports
├── README.md                            # This file
├── openenv.yaml                         # OpenEnv manifest (6 tasks)
├── pyproject.toml                       # Project metadata & dependencies
├── .dockerignore                        # Docker build exclusions
├── Dockerfile                           # Container image definition
├── client.py                            # TrafficControlEnv HTTP/WS client
├── models.py                            # Action, Observation, State models
├── inference.py                         # LLM agent inference script
├── test_benchmark.py                    # Oracle validation suite
└── server/
    ├── __init__.py                      # Server module exports
    ├── app.py                           # FastAPI application
    ├── scenarios.py                     # Scenario definitions (6 scenarios)
    ├── tools.py                         # Tool implementations
    └── traffic_control_env_environment.py  # Core environment logic
```

---

## Environment Design Notes

- **Traffic simulation**: Each step, queues drain by 3 vehicles in the green direction and grow by 1 in the red direction.
- **Concurrent sessions**: The server supports up to 100 simultaneous WebSocket sessions (`SUPPORTS_CONCURRENT_SESSIONS = True`).
- **Deterministic evaluation**: Each scenario has fixed initial conditions, making oracle validation reproducible.
- **Resolution detection**: Episodes end early (`resolved=True`) when the agent sends a message containing the correct scenario keywords AND the total queue has dropped below the difficulty-scaled threshold.
