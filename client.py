# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for the OpenEnv Content Moderation Environment.

This follows the standard OpenEnv typed client pattern:
- subclass EnvClient[Action, Observation, State]
- implement _step_payload(...)
- implement _parse_result(...)
- implement _parse_state(...)
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import ModerationAction, ModerationObservation, ModerationState
except ImportError:
    from models import ModerationAction, ModerationObservation, ModerationState


class ContentModerationEnv(
    EnvClient[ModerationAction, ModerationObservation, ModerationState]
):
    """
    Client for the content moderation benchmark.

    Example:
        async with ContentModerationEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(
                ModerationAction(
                    operation="finalize",
                    violation_type="safe",
                    severity="none",
                    enforcement="allow",
                    escalate=False,
                    rationale="Allowed criticism with no target-specific abuse."
                )
            )

    Sync usage is also supported by OpenEnv's .sync() wrapper.
    """

    def _step_payload(self, action: ModerationAction) -> Dict:
        """Convert action into a JSON payload for HTTP/WebSocket transport."""
        return {
            "operation": action.operation,
            "requested_context_keys": action.requested_context_keys,
            "violation_type": action.violation_type,
            "severity": action.severity,
            "enforcement": action.enforcement,
            "escalate": action.escalate,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ModerationObservation]:
        """Parse a server response into StepResult[ModerationObservation]."""
        obs_data = payload.get("observation", {})
        observation = ModerationObservation(
            benchmark=obs_data.get("benchmark", "openenv_content_moderation"),
            task=obs_data.get("task", "task_easy"),
            case_id=obs_data.get("case_id", ""),
            title=obs_data.get("title", ""),
            instructions=obs_data.get("instructions", ""),
            content_type=obs_data.get("content_type", ""),
            content_text=obs_data.get("content_text", ""),
            report_reason=obs_data.get("report_reason", ""),
            user_metadata=obs_data.get("user_metadata", {}),
            policy_snippets=obs_data.get("policy_snippets", []),
            confidence_hint=obs_data.get("confidence_hint", ""),
            conversation_excerpt=obs_data.get("conversation_excerpt"),
            moderator_note=obs_data.get("moderator_note"),
            risk_indicator=obs_data.get("risk_indicator"),
            prior_content_pattern=obs_data.get("prior_content_pattern"),
            available_actions=obs_data.get("available_actions", []),
            previously_requested_context=obs_data.get("previously_requested_context", []),
            last_action_error=obs_data.get("last_action_error"),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 0),
            current_score=obs_data.get("current_score", 0.0),
            verifier_summary=obs_data.get("verifier_summary", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ModerationState:
        """Parse server state payload into the typed ModerationState."""
        return ModerationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            active_task=payload.get("active_task", "task_easy"),
            case_id=payload.get("case_id"),
            done=payload.get("done", False),
            requested_context_keys=payload.get("requested_context_keys", []),
            final_score=payload.get("final_score", 0.0),
            final_decision=payload.get("final_decision", {}),
            task_case_index=payload.get("task_case_index", 0),
        )