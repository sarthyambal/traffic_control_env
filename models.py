# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Control Env Environment.

Defines the Action, Observation, and State types for an autonomous traffic
control simulator where LLM agents manage city intersections, respond to
accidents, and route emergency vehicles.
"""

from typing import Optional, Dict, Any, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator
import json

class TrafficControlAction(Action):
    """Action for the Traffic Control Env environment."""
    tool_name: Optional[str] = Field(None, description="Name of the tool to use")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    message: Optional[str] = Field(None, description="Optional reasoning or summary message")

    @field_validator("tool_args", mode="before")
    @classmethod
    def parse_tool_args(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v

class TrafficControlObservation(Observation):
    """Observation from the Traffic Control Env environment."""
    done: bool = Field(default=False)
    reward: float = Field(default=0.01)  # must be strictly > 0; never use 0.0 (Phase 2 validator rejects it)
    customer_query: str = Field(default="")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Result of the tool execution")
    feedback: str = Field(default="", description="System feedback")
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)
    scenario_id: str = Field(default="")
    difficulty: str = Field(default="")
    steps_taken: int = Field(default=0)
    max_steps: int = Field(default=15)

class TrafficControlState(State):
    """Internal state."""
    scenario_id: str = "easy_rush_hour"
    difficulty: str = "easy"
    partial_score: float = 0.05  # safe default strictly inside (0, 1)
    resolved: bool = False
    escalated: bool = False  # FIX Bug 5: field was missing, referenced in client.py _parse_state
    tools_called: List[str] = Field(default_factory=list)
