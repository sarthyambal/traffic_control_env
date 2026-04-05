from typing import Dict, Any, List
import copy

AVAILABLE_TOOLS = [
    "get_network_status",
    "get_intersection_status",
    "set_traffic_light",
    "dispatch_emergency_vehicle",
    "get_traffic_prediction",
]

TOOL_DESCRIPTIONS = {
    "get_network_status": {
        "name": "get_network_status",
        "description": "Get a high-level overview of all intersections — identifies accidents, emergencies, severe gridlock, and inter-intersection connections.",
        "parameters": {},
    },
    "get_intersection_status": {
        "name": "get_intersection_status",
        "description": "Get detailed queue lengths, light phase, and neighbour connections for a specific intersection.",
        "parameters": {
            "intersection_id": "string - The ID of the intersection (e.g., INT-1)",
        },
    },
    "set_traffic_light": {
        "name": "set_traffic_light",
        "description": "Change the light phase of an intersection. NOTE: changing a phase causes cars from that direction to begin flowing into connected neighbours — plan the sequence carefully.",
        "parameters": {
            "intersection_id": "string - The ID of the intersection",
            "phase": "string - 'NS_GREEN' (North-South green) or 'EW_GREEN' (East-West green)",
        },
    },
    "dispatch_emergency_vehicle": {
        "name": "dispatch_emergency_vehicle",
        "description": "Dispatch emergency services to clear an accident at a specific intersection. Must be called before traffic can flow through that intersection.",
        "parameters": {
            "intersection_id": "string - ID of the intersection with the accident",
        },
    },
    "get_traffic_prediction": {
        "name": "get_traffic_prediction",
        "description": (
            "Simulate the city grid N steps into the future using the current light phases "
            "and return the predicted queue states. Use this to evaluate whether a proposed "
            "sequence of light changes will resolve congestion before committing."
        ),
        "parameters": {
            "steps_ahead": "integer - Number of simulation steps to forecast (1-10)",
        },
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _simulate_one_step(grid: Dict[str, Any]) -> None:
    """Single-step deterministic simulation with inter-intersection transfer."""
    DRAIN = 3
    BUILD = 1
    MAX_Q = 60

    transfers: Dict[str, Dict[str, int]] = {}

    for i_id, data in grid.items():
        if data.get("has_accident"):
            continue

        phase = data["light_phase"]
        queue = data["queue"]
        conns = data.get("connections", {})

        def _conn_blocked(direction: str) -> bool:
            nb = conns.get(direction)
            return bool(nb and grid.get(nb, {}).get("has_accident"))

        def _drain(direction: str, opposite_nb_dir: str) -> int:
            if _conn_blocked(direction):
                return 0
            drained = min(queue[direction], DRAIN)
            queue[direction] = max(0, queue[direction] - drained)
            nb = conns.get(direction)
            if nb and drained > 0:
                amt = max(1, drained // 2)
                transfers.setdefault(nb, {"N": 0, "S": 0, "E": 0, "W": 0})
                transfers[nb][opposite_nb_dir] += amt
            return drained

        def _build(direction: str) -> None:
            queue[direction] = min(queue[direction] + BUILD, MAX_Q)

        if phase == "NS_GREEN":
            _drain("N", "S")
            _drain("S", "N")
            _build("E")
            _build("W")
        elif phase == "EW_GREEN":
            _drain("E", "W")
            _drain("W", "E")
            _build("N")
            _build("S")

    # Apply inter-intersection transfers
    for target, amounts in transfers.items():
        if target in grid and not grid[target].get("has_accident"):
            tq = grid[target]["queue"]
            for d, amt in amounts.items():
                tq[d] = min(tq[d] + amt, MAX_Q)


# ── tool implementations ──────────────────────────────────────────────────────

def get_network_status(city_grid: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    overview = {}
    for i_id, data in city_grid.items():
        total_q = sum(data["queue"].values())
        if data.get("has_accident"):
            status = "ACCIDENT"
        elif data.get("has_emergency_vehicle"):
            status = "EMERGENCY_VEHICLE"
        elif total_q >= 40:
            status = "SEVERE_GRIDLOCK"
        elif total_q >= 20:
            status = "HEAVY_TRAFFIC"
        else:
            status = "normal"

        conns = {d: nb for d, nb in data.get("connections", {}).items() if nb}
        entry = {
            "status": status,
            "total_vehicles_waiting": total_q,
            "light_phase": data["light_phase"],
        }
        if conns:
            entry["connected_to"] = conns
        overview[i_id] = entry

    return {"success": True, "network": overview}


def get_intersection_status(city_grid: Dict[str, Any], intersection_id: str = "", **kwargs) -> Dict[str, Any]:
    if not intersection_id:
        return {"success": False, "error": "intersection_id is required"}
    if intersection_id not in city_grid:
        return {"success": False, "error": f"Unknown intersection '{intersection_id}'"}

    data = city_grid[intersection_id]
    result: Dict[str, Any] = {
        "success": True,
        "intersection_id": intersection_id,
        "light_phase": data["light_phase"],
        "queue": data["queue"],
        "total_vehicles": sum(data["queue"].values()),
        "has_accident": data.get("has_accident", False),
        "has_emergency_vehicle": data.get("has_emergency_vehicle", False),
    }
    conns = {d: nb for d, nb in data.get("connections", {}).items() if nb}
    if conns:
        result["connections"] = conns
    if data.get("emergency_direction"):
        result["emergency_direction"] = data["emergency_direction"]
    return result


def set_traffic_light(city_grid: Dict[str, Any], intersection_id: str = "", phase: str = "", **kwargs) -> Dict[str, Any]:
    if not intersection_id or not phase:
        return {"success": False, "error": "intersection_id and phase are required"}
    if intersection_id not in city_grid:
        return {"success": False, "error": f"Unknown intersection '{intersection_id}'"}
    if phase not in ("NS_GREEN", "EW_GREEN"):
        return {"success": False, "error": "phase must be 'NS_GREEN' or 'EW_GREEN'"}
    if city_grid[intersection_id].get("has_accident"):
        return {"success": False, "error": f"Cannot change light at {intersection_id} — accident present. Dispatch first."}

    city_grid[intersection_id]["light_phase"] = phase
    conns = {d: nb for d, nb in city_grid[intersection_id].get("connections", {}).items() if nb}
    msg = f"Light at {intersection_id} set to {phase}."
    if conns:
        msg += f" Connected to: {conns}. Flow changes will cascade to neighbours next step."
    return {"success": True, "intersection_id": intersection_id, "new_phase": phase, "message": msg}


def dispatch_emergency_vehicle(city_grid: Dict[str, Any], intersection_id: str = "", **kwargs) -> Dict[str, Any]:
    if not intersection_id:
        return {"success": False, "error": "intersection_id is required"}
    if intersection_id not in city_grid:
        return {"success": False, "error": f"Unknown intersection '{intersection_id}'"}
    if not city_grid[intersection_id].get("has_accident"):
        return {"success": False, "error": f"No accident at {intersection_id}."}

    city_grid[intersection_id]["has_accident"] = False
    city_grid[intersection_id]["has_emergency_vehicle"] = False
    return {
        "success": True,
        "intersection_id": intersection_id,
        "message": f"Emergency services dispatched to {intersection_id}. Accident cleared — traffic can now flow.",
    }


def get_traffic_prediction(city_grid: Dict[str, Any], steps_ahead: int = 3, **kwargs) -> Dict[str, Any]:
    """Deterministic lookahead: simulate N steps on a deep copy and report predicted queues."""
    steps_ahead = max(1, min(int(steps_ahead), 10))
    grid_copy = copy.deepcopy(city_grid)

    for _ in range(steps_ahead):
        _simulate_one_step(grid_copy)

    prediction = {}
    for i_id, data in grid_copy.items():
        prediction[i_id] = {
            "predicted_queue": data["queue"],
            "predicted_total": sum(data["queue"].values()),
            "light_phase": data["light_phase"],
        }

    return {
        "success": True,
        "steps_ahead": steps_ahead,
        "prediction": prediction,
        "tip": "Use this to evaluate whether current light phases will clear congestion before committing to changes.",
    }


# ── registry ─────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "get_network_status": get_network_status,
    "get_intersection_status": get_intersection_status,
    "set_traffic_light": set_traffic_light,
    "dispatch_emergency_vehicle": dispatch_emergency_vehicle,
    "get_traffic_prediction": get_traffic_prediction,
}


def call_tool(tool_name: str, tool_args: Dict[str, Any], city_grid: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"Unknown tool '{tool_name}'. Available: {AVAILABLE_TOOLS}"}
    try:
        return TOOL_REGISTRY[tool_name](city_grid=city_grid, **tool_args)
    except Exception as e:
        return {"success": False, "error": f"Execution error in '{tool_name}': {e}"}
