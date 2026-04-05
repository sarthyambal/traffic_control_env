"""
Oracle Benchmark Suite
===================================
Hardcoded perfect-play workflows for all 6 scenarios.
Verifies that each scenario is solvable and scores >= 0.8.
"""

from typing import Any, Dict, List
from server.traffic_control_env_environment import TrafficControlEnvironment
from models import TrafficControlAction, TrafficControlObservation


class OracleAgent:
    def __init__(self, actions: List[TrafficControlAction]):
        self.actions = actions
        self.idx = 0

    def step(self, obs: TrafficControlObservation) -> TrafficControlAction:
        if self.idx < len(self.actions):
            act = self.actions[self.idx]
            self.idx += 1
            return act
        # Fallback: keep sending resolution message after planned actions
        return TrafficControlAction(tool_name=None, message="resolved cleared")


ORACLE_WORKFLOWS = {
    # ── Easy ─────────────────────────────────────────────────────────────────
    "easy_rush_hour": [
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-1"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-1", "phase": "NS_GREEN"},
            message="cleared green north south",
        ),
    ],

    # ── Medium ───────────────────────────────────────────────────────────────
    "medium_accident_response": [
        TrafficControlAction(tool_name="get_network_status", tool_args={}),
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-2"}),
        TrafficControlAction(tool_name="dispatch_emergency_vehicle", tool_args={"intersection_id": "INT-2"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-2", "phase": "EW_GREEN"},
            message="accident cleared dispatched emergency vehicle",
        ),
    ],

    # ── Hard: sequential gridlock ─────────────────────────────────────────────
    "hard_gridlock_unbraiding": [
        TrafficControlAction(tool_name="get_network_status", tool_args={}),
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-1"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-1", "phase": "EW_GREEN"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-2", "phase": "NS_GREEN"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-3", "phase": "EW_GREEN"},
            message="gridlock cleared resolved all intersections",
        ),
    ],

    # ── Hard: emergency corridor ──────────────────────────────────────────────
    "hard_emergency_routing": [
        TrafficControlAction(tool_name="get_network_status", tool_args={}),
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-1"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-1", "phase": "EW_GREEN"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-2", "phase": "EW_GREEN"},
            message="corridor ambulance passed green route clear",
        ),
    ],

    # ── Expert: cascade emergency (3-intersection chain blocked by accident) ──
    "expert_cascade_emergency": [
        TrafficControlAction(tool_name="get_network_status", tool_args={}),
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-2"}),
        TrafficControlAction(tool_name="dispatch_emergency_vehicle", tool_args={"intersection_id": "INT-2"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-2", "phase": "EW_GREEN"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-3", "phase": "EW_GREEN"},
            message="cascade cleared dispatched corridor accident emergency resolved",
        ),
    ],

    # ── Expert: ring deadlock (4-intersection ring, all phased wrong) ─────────
    # Correct unlock order: fix INT-2 and INT-4 first (they receive overflow),
    # then INT-1 and INT-3 (they send the overflow).
    "expert_ring_deadlock": [
        TrafficControlAction(tool_name="get_network_status", tool_args={}),
        TrafficControlAction(tool_name="get_intersection_status", tool_args={"intersection_id": "INT-1"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-2", "phase": "NS_GREEN"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-4", "phase": "NS_GREEN"}),
        TrafficControlAction(tool_name="set_traffic_light", tool_args={"intersection_id": "INT-1", "phase": "EW_GREEN"}),
        TrafficControlAction(
            tool_name="set_traffic_light",
            tool_args={"intersection_id": "INT-3", "phase": "EW_GREEN"},
            message="ring deadlock cleared resolved unbraided all intersections",
        ),
    ],
}


def run_tests():
    env = TrafficControlEnvironment()
    scenarios = list(ORACLE_WORKFLOWS.keys())
    total_score = 0.0

    print("=== OpenEnv Oracle Test Suite ===")
    print(f"Testing {len(scenarios)} scenarios.\n")

    for scenario_id in scenarios:
        obs = env.reset(scenario_id=scenario_id)
        oracle = OracleAgent(ORACLE_WORKFLOWS[scenario_id])

        print(f"\n[{scenario_id.upper()}] ({env._scenario.difficulty})")

        while not obs.done:
            action = oracle.step(obs)
            label = f"Tool={action.tool_name}({action.tool_args})" if action.tool_name else f"Msg='{action.message}'"
            print(f"  → {label}")
            obs = env.step(action)

        final_score = env.state.partial_score
        total_score += final_score

        print(f"  Score: {final_score:.2f}/1.00  |  Steps: {env.state.step_count}  |  Resolved: {env.state.resolved}")
        assert final_score >= 0.8, f"Score too low for oracle in '{scenario_id}' — got {final_score:.2f}"

    avg = total_score / len(scenarios)
    print(f"\n=== Final Avg Score: {avg:.2f} / 1.00 ===")
    assert avg >= 0.90, f"Average score {avg:.2f} below 0.90 target!"
    print("All assertions passed ✓")


if __name__ == "__main__":
    run_tests()
