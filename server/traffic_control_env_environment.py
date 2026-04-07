import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import TrafficControlAction, TrafficControlObservation, TrafficControlState
    from .scenarios import SCENARIOS, get_scenario, Scenario
    from .tools import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS, call_tool, _simulate_one_step
except (ImportError, ModuleNotFoundError):
    from models import TrafficControlAction, TrafficControlObservation, TrafficControlState
    from server.scenarios import SCENARIOS, get_scenario, Scenario
    from server.tools import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS, call_tool, _simulate_one_step


class TrafficControlEnvironment(Environment):
    """
    Autonomous Traffic Control Environment.

    Features:
    - 4 standard scenarios (easy → hard) + 2 expert scenarios
    - Inter-intersection car flow: when one intersection drains its queue, half
      those cars cascade to the connected neighbour — creating realistic spillback.
    - Per-scenario resolution thresholds: expert scenarios require clearing more cars.
    - Generalised ordinal bonuses for set_traffic_light (supports up to N calls).
    - Accident blocking: lights cannot change and cars cannot flow through blocked intersections.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 20

    def __init__(self):
        super().__init__()
        self._state = TrafficControlState()
        self._scenario: Optional[Scenario] = None
        self._city_grid: Dict[str, Any] = {}
        self._tools_called: List[str] = []
        self._tool_calls_detail: List[Dict[str, Any]] = []
        self._total_reward: float = 0.0
        self._resolved: bool = False

    # ── public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: str = "easy_rush_hour",
        **kwargs: Any,
    ) -> TrafficControlObservation:
        self._scenario = get_scenario(scenario_id)
        self._city_grid = copy.deepcopy(self._scenario.city_grid)
        self._tools_called = []
        self._tool_calls_detail = []
        self._total_reward = 0.0
        self._resolved = False

        ep_id = episode_id or str(uuid4())
        self._state = TrafficControlState(
            episode_id=ep_id,
            step_count=0,
            scenario_id=scenario_id,
            difficulty=self._scenario.difficulty,
        )

        return TrafficControlObservation(
            done=False,
            reward=0.0,
            customer_query=self._scenario.description,
            tool_result=None,
            available_tools=list(TOOL_DESCRIPTIONS.values()),
            scenario_id=scenario_id,
            difficulty=self._scenario.difficulty,
            feedback=f"Episode started: {self._scenario.description}",
            steps_taken=0,
            max_steps=self.MAX_STEPS,
        )

    def step(
        self,
        action: TrafficControlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficControlObservation:
        self._state.step_count += 1
        step_reward = 0.0
        tool_result = None
        feedback_parts: List[str] = []

        # ── traffic physics (runs every step, before the agent's action)
        _simulate_one_step(self._city_grid)

        if action.tool_name:
            # Count BEFORE appending (0-based index for ordinal bonus calculation)
            tool_call_ordinal = self._tools_called.count(action.tool_name)

            tool_result = call_tool(action.tool_name, action.tool_args, self._city_grid)
            self._tools_called.append(action.tool_name)

            call_sig = {"tool": action.tool_name, "args": dict(sorted(action.tool_args.items()))}
            is_duplicate = call_sig in self._tool_calls_detail
            self._tool_calls_detail.append(call_sig)

            if is_duplicate:
                step_reward += self._scenario.penalty_repeated_tool
                feedback_parts.append(f"⚠ Duplicate call penalty ({action.tool_name}): {self._scenario.penalty_repeated_tool:+.2f}")
            else:
                tool_reward = self._compute_tool_reward(action.tool_name, action.tool_args, tool_result, tool_call_ordinal)
                step_reward += tool_reward
                if tool_reward > 0:
                    feedback_parts.append(f"✓ Good action +{tool_reward:.2f}")
                elif tool_reward < 0:
                    feedback_parts.append(f"✗ Wrong tool {tool_reward:.2f}")

        if action.message:
            msg_reward = self._compute_message_reward(action.message)
            step_reward += msg_reward
            if msg_reward > 0:
                feedback_parts.append(f"✓ Resolution analysis +{msg_reward:.2f}")

        done = self._check_done()

        self._total_reward += step_reward
        self._total_reward = max(self._total_reward, -0.5)
        self._state.partial_score = min(max(self._total_reward, 0.01), 0.99)

        if done:
            final = min(max(self._total_reward, 0.01), 0.99)
            feedback_parts.append(f"Episode complete. Final score: {final:.3f}/1.000")

        return TrafficControlObservation(
            done=done,
            reward=round(self._safe_reward(step_reward), 4),
            customer_query=self._scenario.description,
            tool_result=tool_result,
            available_tools=list(TOOL_DESCRIPTIONS.values()),
            scenario_id=self._state.scenario_id,
            difficulty=self._state.difficulty,
            feedback=" | ".join(feedback_parts) if feedback_parts else "Step simulated.",
            steps_taken=self._state.step_count,
            max_steps=self.MAX_STEPS,
        )

    @property
    def state(self) -> TrafficControlState:
        self._state.resolved = self._resolved
        self._state.tools_called = list(self._tools_called)
        return self._state

    # ── reward helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_reward(r: float) -> float:
        """Clamp a step reward to the strict open interval (0.01, 0.99).

        Prevents invalid 0.0 or negative rewards from being returned to
        OpenEnv, which enforces strictly-bounded task scores.
        """
        return max(0.01, min(0.99, r))

    def _compute_tool_reward(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Dict[str, Any],
        call_ordinal: int,  # 0-indexed count of this tool BEFORE this call
    ) -> float:
        if not self._scenario:
            return 0.0

        scenario = self._scenario

        if tool_name not in scenario.correct_tool_sequence:
            return scenario.penalty_wrong_tool

        reward = 0.0

        if tool_name == "set_traffic_light":
            intersection_id = tool_args.get("intersection_id", "")
            q = self._city_grid.get(intersection_id, {}).get("queue", {"N": 0, "S": 0, "E": 0, "W": 0})
            base = scenario.partial_rewards.get("set_traffic_light", 0.1)
            phase = tool_args.get("phase", "")

            # Base reward only if it was the optimal direction
            if phase == "NS_GREEN" and (q["N"] > q["E"] or q["S"] > q["W"]):
                reward = base
            elif phase == "EW_GREEN" and (q["E"] > q["N"] or q["W"] > q["S"]):
                reward = base

            # Generalised ordinal bonus: set_traffic_light_1, _2, _3, _4, …
            # call_ordinal is 0-based (first call → ordinal=0 → key _1)
            ordinal_key = f"set_traffic_light_{call_ordinal + 1}"
            reward += scenario.partial_rewards.get(ordinal_key, 0.0)

        else:
            reward = scenario.partial_rewards.get(tool_name, 0.0)

        return reward

    def _compute_message_reward(self, message: str) -> float:
        if not self._scenario:
            return 0.0

        msg_lower = message.lower()
        kw_list = self._scenario.resolution_keywords
        if not kw_list:
            return 0.0

        hits = sum(1 for kw in kw_list if kw.lower() in msg_lower)
        ratio = hits / len(kw_list)

        if hits == 0:
            return 0.0

        reward = ratio * 0.2

        if ratio >= 0.5:
            total_q = sum(sum(d["queue"].values()) for d in self._city_grid.values())
            if total_q < self._scenario.resolution_threshold:
                self._resolved = True

        return reward

    def _check_done(self) -> bool:
        return self._state.step_count >= self.MAX_STEPS or self._resolved
