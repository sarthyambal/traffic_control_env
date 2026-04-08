# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Traffic Control Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TrafficControlAction, TrafficControlObservation, TrafficControlState
except ImportError:
    from models import TrafficControlAction, TrafficControlObservation, TrafficControlState


class TrafficControlEnv(
    EnvClient[TrafficControlAction, TrafficControlObservation, TrafficControlState]
):
    """
    Client for the Traffic Control Env Environment.
    """

    def _step_payload(self, action: TrafficControlAction) -> Dict:
        """
        Convert TrafficControlAction to JSON payload for step message.
        """
        return {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficControlObservation]:
        """
        Parse server response into StepResult[TrafficControlObservation].
        """
        obs_data = payload.get("observation", {})
        observation = TrafficControlObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.01),  # never default to 0.0 — Phase 2 validator rejects it
            customer_query=obs_data.get("customer_query", ""),
            tool_result=obs_data.get("tool_result"),
            feedback=obs_data.get("feedback", ""),
            available_tools=obs_data.get("available_tools", []),
            scenario_id=obs_data.get("scenario_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 15),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> TrafficControlState:
        """
        Parse server response into State object.
        """
        return TrafficControlState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            scenario_id=payload.get("scenario_id", ""),
            difficulty=payload.get("difficulty", ""),
            partial_score=payload.get("partial_score", 0.05),  # never default to 0.0
            resolved=payload.get("resolved", False),
            escalated=payload.get("escalated", False),
            tools_called=payload.get("tools_called", [])
        )
