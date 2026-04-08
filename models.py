# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the OpenEnv Content Moderation Environment.

This benchmark simulates a production-style moderation queue. The agent can:
- review a case
- request additional context
- finalize a moderation decision

The models below intentionally stay close to OpenEnv's typed Action /
Observation / State style while carrying enough structured information for
deterministic grading.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

TaskName = Literal["task_easy", "task_medium", "task_hard"]
OperationName = Literal["review", "request_context", "finalize"]
ViolationName = Literal[
    "safe",
    "spam",
    "phishing",
    "harassment",
    "hate",
    "self_harm",
    "sexual",
    "violence",
    "other",
]
SeverityName = Literal["none", "low", "medium", "high", "critical"]
EnforcementName = Literal["allow", "limit", "remove", "suspend", "escalate_only"]
ContextKey = Literal[
    "conversation_excerpt",
    "moderator_note",
    "risk_indicator",
    "prior_content_pattern",
]


class ModerationAction(Action):
    """Typed action for the moderation environment."""

    operation: OperationName = Field(
        ...,
        description="review, request_context, or finalize",
    )
    requested_context_keys: List[ContextKey] = Field(
        default_factory=list,
        description="Optional hidden context fields to reveal",
    )
    violation_type: Optional[ViolationName] = Field(
        default=None,
        description="Predicted policy class",
    )
    severity: Optional[SeverityName] = Field(
        default=None,
        description="Predicted severity level",
    )
    enforcement: Optional[EnforcementName] = Field(
        default=None,
        description="Chosen enforcement action",
    )
    escalate: Optional[bool] = Field(
        default=None,
        description="Whether the case should escalate to human review / crisis queue",
    )
    rationale: str = Field(
        default="",
        max_length=400,
        description="Short single-line rationale",
    )

    @field_validator("requested_context_keys")
    @classmethod
    def _dedupe_context_keys(cls, keys: List[str]) -> List[str]:
        deduped: List[str] = []
        for key in keys:
            if key not in deduped:
                deduped.append(key)
        return deduped


class ModerationObservation(Observation):
    """Typed observation returned by the moderation environment."""

    benchmark: str = Field(
        default="openenv_content_moderation",
        description="Benchmark identifier",
    )
    task: TaskName = Field(..., description="Current task tier")
    case_id: str = Field(..., description="Current moderation case ID")
    title: str = Field(..., description="Short case title")

    instructions: str = Field(..., description="Current task instructions")
    content_type: str = Field(..., description="Type of content under review")
    content_text: str = Field(..., description="Main content text")
    report_reason: str = Field(..., description="Why the case was reported")

    user_metadata: Dict = Field(
        default_factory=dict,
        description="Author metadata and account history",
    )
    policy_snippets: List[Dict] = Field(
        default_factory=list,
        description="Policy excerpts available to the agent",
    )
    confidence_hint: str = Field(
        default="",
        description="Heuristic note from tooling",
    )

    conversation_excerpt: Optional[str] = Field(default=None)
    moderator_note: Optional[str] = Field(default=None)
    risk_indicator: Optional[str] = Field(default=None)
    prior_content_pattern: Optional[str] = Field(default=None)

    available_actions: List[OperationName] = Field(default_factory=list)
    previously_requested_context: List[ContextKey] = Field(default_factory=list)
    last_action_error: Optional[str] = Field(default=None)

    step_count: int = Field(default=0, description="Current step count")
    max_steps: int = Field(default=0, description="Max allowed steps")

    current_score: float = Field(
        default=0.0,
        description="Final normalized score when done, else 0.0",
    )
    verifier_summary: Dict = Field(
        default_factory=dict,
        description="Bounded deterministic verifier summary",
    )


class ModerationState(State):
    """OpenEnv state model for the content moderation environment."""

    active_task: TaskName = Field(default="task_easy")
    case_id: Optional[str] = Field(default=None)
    done: bool = Field(default=False)
    requested_context_keys: List[ContextKey] = Field(default_factory=list)
    final_score: float = Field(default=0.0)
    final_decision: Dict = Field(default_factory=dict)
    task_case_index: int = Field(default=0)