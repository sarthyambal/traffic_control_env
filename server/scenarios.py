from typing import Dict, Any, List, Optional


class Scenario:
    def __init__(
        self,
        scenario_id: str,
        difficulty: str,
        description: str,
        city_grid: Dict[str, Any],
        correct_tool_sequence: List[str],
        resolution_keywords: Optional[List[str]] = None,
        partial_rewards: Optional[Dict[str, float]] = None,
        penalty_wrong_tool: float = -0.05,
        penalty_repeated_tool: float = -0.03,
        resolution_threshold: int = 20,  # total queue must be below this to resolve
    ):
        self.scenario_id = scenario_id
        self.difficulty = difficulty
        self.description = description
        self.city_grid = city_grid
        self.correct_tool_sequence = correct_tool_sequence
        self.resolution_keywords = resolution_keywords or []
        self.partial_rewards = partial_rewards or {}
        self.penalty_wrong_tool = penalty_wrong_tool
        self.penalty_repeated_tool = penalty_repeated_tool
        self.resolution_threshold = resolution_threshold


SCENARIOS = {
    # ─────────────────────────────────────────────
    # EASY: single intersection, clear the queue
    # ─────────────────────────────────────────────
    "easy_rush_hour": Scenario(
        scenario_id="easy_rush_hour",
        difficulty="easy",
        description="Heavy North-South traffic at INT-1. Assess the situation and set the correct light phase to clear the queue.",
        city_grid={
            "INT-1": {
                "light_phase": "EW_GREEN",
                "queue": {"N": 8, "S": 6, "E": 2, "W": 1},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {},
            }
        },
        correct_tool_sequence=["get_intersection_status", "set_traffic_light"],
        resolution_keywords=["cleared", "green", "north", "south"],
        partial_rewards={
            "get_intersection_status": 0.2,
            "set_traffic_light": 0.5,
            "correct_sequence": 0.3,
        },
        resolution_threshold=20,
    ),

    # ─────────────────────────────────────────────
    # MEDIUM: accident + two-tool response
    # ─────────────────────────────────────────────
    "medium_accident_response": Scenario(
        scenario_id="medium_accident_response",
        difficulty="medium",
        description="Accident at INT-2. Dispatch emergency services then restore traffic flow.",
        city_grid={
            "INT-2": {
                "light_phase": "NS_GREEN",
                "queue": {"N": 5, "S": 20, "E": 10, "W": 8},
                "has_accident": True,
                "has_emergency_vehicle": False,
                "connections": {},
            }
        },
        correct_tool_sequence=[
            "get_network_status", "get_intersection_status",
            "dispatch_emergency_vehicle", "set_traffic_light",
        ],
        resolution_keywords=["accident", "cleared", "dispatched"],
        partial_rewards={
            "get_network_status": 0.1,
            "get_intersection_status": 0.1,
            "dispatch_emergency_vehicle": 0.4,
            "set_traffic_light": 0.3,
            "correct_sequence": 0.1,
        },
        resolution_threshold=50,
    ),

    # ─────────────────────────────────────────────
    # HARD: 3-intersection sequential gridlock
    # ─────────────────────────────────────────────
    "hard_gridlock_unbraiding": Scenario(
        scenario_id="hard_gridlock_unbraiding",
        difficulty="hard",
        description="Gridlock across INT-1, INT-2, and INT-3. Sequentially clear each intersection in the correct order.",
        city_grid={
            "INT-1": {"light_phase": "NS_GREEN", "queue": {"N": 20, "S": 0, "E": 15, "W": 0}, "has_accident": False, "has_emergency_vehicle": False, "connections": {}},
            "INT-2": {"light_phase": "EW_GREEN", "queue": {"N": 0, "S": 20, "E": 0, "W": 15}, "has_accident": False, "has_emergency_vehicle": False, "connections": {}},
            "INT-3": {"light_phase": "NS_GREEN", "queue": {"N": 18, "S": 18, "E": 0, "W": 0}, "has_accident": False, "has_emergency_vehicle": False, "connections": {}},
        },
        correct_tool_sequence=[
            "get_network_status", "get_intersection_status",
            "set_traffic_light", "set_traffic_light", "set_traffic_light",
        ],
        resolution_keywords=["gridlock", "cleared", "resolved"],
        partial_rewards={
            "get_network_status": 0.1,
            "get_intersection_status": 0.2,
            "set_traffic_light_1": 0.2,
            "set_traffic_light_2": 0.2,
            "set_traffic_light_3": 0.2,
            "correct_sequence": 0.1,
        },
        resolution_threshold=80,
    ),

    # ─────────────────────────────────────────────
    # HARD: emergency corridor across 2 intersections
    # ─────────────────────────────────────────────
    "hard_emergency_routing": Scenario(
        scenario_id="hard_emergency_routing",
        difficulty="hard",
        description="Ambulance at INT-1 must pass through INT-2 going East. Create a green corridor across both intersections.",
        city_grid={
            "INT-1": {"light_phase": "NS_GREEN", "queue": {"N": 5, "S": 2, "E": 15, "W": 2}, "has_accident": False, "has_emergency_vehicle": True, "emergency_direction": "E", "connections": {"E": "INT-2"}},
            "INT-2": {"light_phase": "NS_GREEN", "queue": {"N": 3, "S": 3, "E": 8, "W": 1}, "has_accident": False, "has_emergency_vehicle": False, "emergency_direction": None, "connections": {"W": "INT-1"}},
        },
        correct_tool_sequence=[
            "get_network_status", "get_intersection_status",
            "set_traffic_light", "set_traffic_light",
        ],
        resolution_keywords=["corridor", "ambulance", "passed"],
        partial_rewards={
            "get_network_status": 0.2,
            "set_traffic_light_1": 0.3,
            "set_traffic_light_2": 0.3,
            "correct_sequence": 0.2,
        },
        resolution_threshold=80,
    ),

    # ─────────────────────────────────────────────
    # EXPERT: 3-intersection chained accident cascade
    # Accident at INT-2 blocks the whole E-W corridor.
    # INT-1 and INT-3 back up because their outbound
    # lanes point at a blocked neighbour.
    # ─────────────────────────────────────────────
    "expert_cascade_emergency": Scenario(
        scenario_id="expert_cascade_emergency",
        difficulty="expert",
        description=(
            "3-intersection E-W corridor: INT-1 ↔ INT-2 ↔ INT-3. "
            "Accident at INT-2 has frozen the entire chain — INT-1's East "
            "queue and INT-3's West queue cannot drain. "
            "Dispatch, then restore flow to all three intersections in order."
        ),
        city_grid={
            "INT-1": {
                "light_phase": "EW_GREEN",
                "queue": {"N": 2, "S": 2, "E": 20, "W": 3},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"E": "INT-2"},
            },
            "INT-2": {
                "light_phase": "NS_GREEN",
                "queue": {"N": 8, "S": 8, "E": 12, "W": 15},
                "has_accident": True,
                "has_emergency_vehicle": False,
                "connections": {"W": "INT-1", "E": "INT-3"},
            },
            "INT-3": {
                "light_phase": "NS_GREEN",
                "queue": {"N": 3, "S": 3, "E": 2, "W": 20},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"W": "INT-2"},
            },
        },
        correct_tool_sequence=[
            "get_network_status", "get_intersection_status",
            "dispatch_emergency_vehicle",
            "set_traffic_light", "set_traffic_light",
        ],
        resolution_keywords=["cascade", "cleared", "dispatched", "corridor", "accident"],
        partial_rewards={
            "get_network_status": 0.1,
            "get_intersection_status": 0.1,
            "dispatch_emergency_vehicle": 0.3,
            "set_traffic_light_1": 0.2,
            "set_traffic_light_2": 0.2,
            "correct_sequence": 0.1,
        },
        penalty_wrong_tool=-0.07,
        penalty_repeated_tool=-0.05,
        resolution_threshold=120,
    ),

    # ─────────────────────────────────────────────
    # EXPERT: 4-intersection ring deadlock
    # Every intersection is green in the WRONG phase,
    # and each one feeds overflow into the next.
    # Agent must find the correct unlock order.
    #
    #  INT-4 ── INT-3
    #    |         |
    #  INT-1 ── INT-2
    # ─────────────────────────────────────────────
    "expert_ring_deadlock": Scenario(
        scenario_id="expert_ring_deadlock",
        difficulty="expert",
        description=(
            "4-intersection ring deadlock. Every intersection is phased wrong: "
            "E-W heavy traffic sits at red while N-S trickles through. "
            "Connections: INT-1↔INT-2 (E-W), INT-2↔INT-3 (N-S), "
            "INT-3↔INT-4 (E-W), INT-4↔INT-1 (N-S). "
            "Identify the unlock order — wrong sequence causes cascade overflow into neighbours."
        ),
        city_grid={
            "INT-1": {
                "light_phase": "NS_GREEN",   # wrong — needs EW_GREEN
                "queue": {"N": 3, "S": 3, "E": 20, "W": 18},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"E": "INT-2", "N": "INT-4"},
            },
            "INT-2": {
                "light_phase": "EW_GREEN",   # wrong — needs NS_GREEN
                "queue": {"N": 20, "S": 18, "E": 3, "W": 3},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"W": "INT-1", "N": "INT-3"},
            },
            "INT-3": {
                "light_phase": "NS_GREEN",   # wrong — needs EW_GREEN
                "queue": {"N": 3, "S": 3, "E": 18, "W": 20},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"W": "INT-4", "S": "INT-2"},
            },
            "INT-4": {
                "light_phase": "EW_GREEN",   # wrong — needs NS_GREEN
                "queue": {"N": 18, "S": 20, "E": 3, "W": 3},
                "has_accident": False,
                "has_emergency_vehicle": False,
                "connections": {"E": "INT-3", "S": "INT-1"},
            },
        },
        correct_tool_sequence=[
            "get_network_status", "get_intersection_status",
            "set_traffic_light", "set_traffic_light",
            "set_traffic_light", "set_traffic_light",
        ],
        resolution_keywords=["ring", "deadlock", "cleared", "resolved", "unbraided"],
        partial_rewards={
            "get_network_status": 0.05,
            "get_intersection_status": 0.05,
            "set_traffic_light_1": 0.2,
            "set_traffic_light_2": 0.2,
            "set_traffic_light_3": 0.2,
            "set_traffic_light_4": 0.2,
            "correct_sequence": 0.1,
        },
        penalty_wrong_tool=-0.07,
        penalty_repeated_tool=-0.05,
        resolution_threshold=220,
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    import copy
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        raise ValueError(f"Unknown scenario: '{scenario_id}'. Available: {list(SCENARIOS)}")
    return copy.deepcopy(scenario)
