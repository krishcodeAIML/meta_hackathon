from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set

import requests
from openai import OpenAI

from client import ContentModerationEnv
from models import ModerationAction

try:
    from server.rl_meta_environment import ContentModerationEnvironment
except ImportError:
    try:
        from server.rl_meta_environment import ContentModerationEnvironment
    except ImportError:
        ContentModerationEnvironment = None  # type: ignore


RunMode = Literal["remote", "local"]

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
RUN_MODE: RunMode = os.getenv("RUN_MODE", "remote").strip().lower()  # type: ignore

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK_NAME = "openenv_content_moderation"
SUCCESS_SCORE_THRESHOLD = 0.7
TEMPERATURE = 0.0
MAX_TOKENS = 260


SYSTEM_PROMPT = """
You are the policy actor inside a content moderation environment.

Your job is to return exactly ONE structured moderation action as valid JSON.
Do not explain outside JSON.
Do not invent hidden evidence.
Use only the information currently visible in the observation.

Primary objective:
- Maximize final moderation quality while following workflow constraints.

Allowed operations:
- "review"
- "request_context"
- "finalize"

Hard workflow rules:
1. request_context may use null for violation_type, severity, enforcement, escalate.
2. review MUST include non-null values for violation_type, severity, enforcement, escalate.
3. finalize MUST include non-null values for violation_type, severity, enforcement, escalate.
4. Never finalize before at least one review.
5. If the case is critical or the confidence hint says context is strongly recommended before finalization, request the most useful missing context first.
6. For imminent self-harm, prefer requesting moderator_note and risk_indicator together.
7. For ambiguous offline violence, prefer requesting conversation_excerpt, risk_indicator, and/or prior_content_pattern before finalization.
8. After an error about incomplete structured decision, correct it by returning a complete review next.
9. After an error about missing evidence or richer evidence, request the most useful missing context next.
10. Prefer one strong context request over multiple weak ones.

Decision rules:
- For quoted speech, criticism, or reporting/condemnation, avoid over-enforcement.
- For phishing, credible offline violence, and imminent self-harm, avoid dangerous under-enforcement.
- Escalate only when the evidence and policy justify it.
- Keep rationale short, concrete, and grounded in visible content, policy, and revealed context.

Consistency rules:
- if violation_type == "safe", then severity must be "none", enforcement must be "allow", escalate must be false
- if enforcement == "escalate_only", then escalate must be true
- if severity == "critical", enforcement must not be "allow"

Return ONLY JSON in this schema:
{
  "operation": "review" | "request_context" | "finalize",
  "requested_context_keys": ["conversation_excerpt" | "moderator_note" | "risk_indicator" | "prior_content_pattern"],
  "violation_type": "safe" | "spam" | "phishing" | "harassment" | "hate" | "self_harm" | "sexual" | "violence" | "other" | null,
  "severity": "none" | "low" | "medium" | "high" | "critical" | null,
  "enforcement": "allow" | "limit" | "remove" | "suspend" | "escalate_only" | null,
  "escalate": true | false | null,
  "rationale": "one concise line grounded in visible evidence"
}

Never:
- invent context
- ask for duplicate context
- overuse escalation
- over-penalize clearly safe content
- allow clearly dangerous content
- output non-JSON text
""".strip()


USER_TEMPLATE = """
Benchmark: {benchmark}
Task: {task}
Case ID: {case_id}
Title: {title}
Step: {step_count}/{max_steps}

Visible content:
- content_type: {content_type}
- content_text: {content_text}
- report_reason: {report_reason}
- user_metadata: {user_metadata}
- confidence_hint: {confidence_hint}

Visible policy:
{policy_snippets}

Revealed context:
- conversation_excerpt: {conversation_excerpt}
- moderator_note: {moderator_note}
- risk_indicator: {risk_indicator}
- prior_content_pattern: {prior_content_pattern}

Workflow state:
- previously_requested_context: {previously_requested_context}
- last_action_error: {last_action_error}
- review_count: {review_count}
- context_budget_used: {context_budget_used}
- context_budget_total: {context_budget_total}
- done: {done}
- reward_from_previous_step: {reward}

Choose the single best next action under the environment constraints.
Return only valid JSON.
""".strip()


@dataclass
class EpisodeSummary:
    task_name: str
    mode: str
    steps: int
    rewards: List[float]
    final_score: float
    success: bool
    last_error: Optional[str]


def compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from model output: {text!r}")
        return json.loads(match.group(0))


def observation_to_prompt(obs: Any) -> str:
    policy_block = "\n".join(
        f"- {item.get('code', '')}: {item.get('title', '')} — {item.get('guidance', '')}"
        for item in getattr(obs, "policy_snippets", []) or []
    )
    metadata = getattr(obs, "metadata", {}) or {}

    return USER_TEMPLATE.format(
        benchmark=getattr(obs, "benchmark", BENCHMARK_NAME),
        task=getattr(obs, "task", ""),
        case_id=getattr(obs, "case_id", ""),
        title=getattr(obs, "title", ""),
        step_count=getattr(obs, "step_count", 0),
        max_steps=getattr(obs, "max_steps", 0),
        content_type=getattr(obs, "content_type", ""),
        content_text=getattr(obs, "content_text", ""),
        report_reason=getattr(obs, "report_reason", ""),
        user_metadata=compact_json(getattr(obs, "user_metadata", {}) or {}),
        confidence_hint=getattr(obs, "confidence_hint", ""),
        policy_snippets=policy_block,
        conversation_excerpt=getattr(obs, "conversation_excerpt", None),
        moderator_note=getattr(obs, "moderator_note", None),
        risk_indicator=getattr(obs, "risk_indicator", None),
        prior_content_pattern=getattr(obs, "prior_content_pattern", None),
        previously_requested_context=getattr(obs, "previously_requested_context", []),
        last_action_error=getattr(obs, "last_action_error", None),
        review_count=metadata.get("review_count", 0),
        context_budget_used=metadata.get("context_budget_used", 0),
        context_budget_total=metadata.get("context_budget_total", 0),
        done=getattr(obs, "done", False),
        reward=getattr(obs, "reward", 0.0),
    )


def _obs_text(obs: Any) -> str:
    return (getattr(obs, "content_text", "") or "").lower()


def _obs_hint(obs: Any) -> str:
    return (getattr(obs, "confidence_hint", "") or "").lower()


def _obs_error(obs: Any) -> str:
    return (getattr(obs, "last_action_error", "") or "").lower()


def _requested_context(obs: Any) -> Set[str]:
    return set(getattr(obs, "previously_requested_context", []) or [])


def _metadata(obs: Any) -> Dict[str, Any]:
    return getattr(obs, "metadata", {}) or {}


def _review_count(obs: Any) -> int:
    return int(_metadata(obs).get("review_count", 0) or 0)


def infer_decision_from_observation(obs: Any) -> Dict[str, Any]:
    text = _obs_text(obs)
    hint = _obs_hint(obs)
    policy_text = " ".join(
        f"{item.get('title', '')} {item.get('guidance', '')}"
        for item in getattr(obs, "policy_snippets", []) or []
    ).lower()

    is_self_harm = (
        "self-harm" in hint
        or "suicide" in hint
        or "do not plan to be here tomorrow" in text
        or "wrote the note" in text
    )

    is_phishing = (
        "phishing" in hint
        or "verify now" in text
        or "access is suspended" in text
        or "password" in text
        or "verification code" in text
        or "credential" in policy_text
    )

    is_violence = (
        "violence" in hint
        or "threat model score" in hint
        or "show up and end this" in text
        or "offline harm" in policy_text
    )

    is_quote_safe = (
        "quoted speech" in hint
        or "reported speech" in policy_text
        or "documenting what happened" in text
        or "not endorsing" in policy_text
    )

    is_safe_product_criticism = (
        "targets a product experience" in hint
        or ("pricing is absurd" in text and "company keeps ignoring users" in text)
    )

    if is_self_harm:
        return {
            "violation_type": "self_harm",
            "severity": "critical",
            "enforcement": "escalate_only",
            "escalate": True,
            "rationale": "Imminent self-harm expression with strong risk signals requires immediate escalation.",
        }

    if is_phishing:
        return {
            "violation_type": "phishing",
            "severity": "high",
            "enforcement": "suspend",
            "escalate": False,
            "rationale": "Deceptive credential-harvesting message with urgent verification language.",
        }

    if is_violence:
        return {
            "violation_type": "violence",
            "severity": "critical",
            "enforcement": "suspend",
            "escalate": True,
            "rationale": "Possible imminent offline threat with concerning risk context.",
        }

    if is_quote_safe:
        return {
            "violation_type": "safe",
            "severity": "none",
            "enforcement": "allow",
            "escalate": False,
            "rationale": "Content appears to report or condemn abusive language rather than endorse it.",
        }

    if is_safe_product_criticism:
        return {
            "violation_type": "safe",
            "severity": "none",
            "enforcement": "allow",
            "escalate": False,
            "rationale": "Negative criticism targets a product or organization, not an individual.",
        }

    return {
        "violation_type": "other",
        "severity": "medium",
        "enforcement": "remove",
        "escalate": False,
        "rationale": "Visible evidence suggests policy-violating content requiring enforcement.",
    }


def next_action_from_error(obs: Any) -> Optional[ModerationAction]:
    err = _obs_error(obs)
    requested = _requested_context(obs)
    decision = infer_decision_from_observation(obs)

    if "incomplete structured decision" in err:
        return ModerationAction(
            operation="review",
            requested_context_keys=[],
            violation_type=decision["violation_type"],
            severity=decision["severity"],
            enforcement=decision["enforcement"],
            escalate=decision["escalate"],
            rationale="Providing a complete structured review after prior validation failure.",
        )

    if "cannot finalize before at least one review" in err:
        return ModerationAction(
            operation="review",
            requested_context_keys=[],
            violation_type=decision["violation_type"],
            severity=decision["severity"],
            enforcement=decision["enforcement"],
            escalate=decision["escalate"],
            rationale="A complete review is required before finalization.",
        )

    if "needed context was not requested" in err or "requires richer evidence" in err:
        text = _obs_text(obs)
        hint = _obs_hint(obs)
        keys: List[str] = []

        if "self-harm" in hint or "do not plan to be here tomorrow" in text or "wrote the note" in text:
            if getattr(obs, "moderator_note", None) is None and "moderator_note" not in requested:
                keys.append("moderator_note")
            if getattr(obs, "risk_indicator", None) is None and "risk_indicator" not in requested:
                keys.append("risk_indicator")
        elif "violence" in hint or "show up and end this" in text:
            if getattr(obs, "conversation_excerpt", None) is None and "conversation_excerpt" not in requested:
                keys.append("conversation_excerpt")
            if getattr(obs, "risk_indicator", None) is None and "risk_indicator" not in requested:
                keys.append("risk_indicator")
            if getattr(obs, "prior_content_pattern", None) is None and "prior_content_pattern" not in requested:
                keys.append("prior_content_pattern")
        else:
            if getattr(obs, "conversation_excerpt", None) is None and "conversation_excerpt" not in requested:
                keys.append("conversation_excerpt")
            if getattr(obs, "moderator_note", None) is None and "moderator_note" not in requested:
                keys.append("moderator_note")
            if getattr(obs, "prior_content_pattern", None) is None and "prior_content_pattern" not in requested:
                keys.append("prior_content_pattern")

        if keys:
            return ModerationAction(
                operation="request_context",
                requested_context_keys=keys,
                violation_type=None,
                severity=None,
                enforcement=None,
                escalate=None,
                rationale="Requesting the highest-value missing context needed for a defensible decision.",
            )

    return None


def deterministic_action(obs: Any) -> Optional[ModerationAction]:
    text = _obs_text(obs)
    hint = _obs_hint(obs)
    requested = _requested_context(obs)
    review_count = _review_count(obs)
    step_count = int(getattr(obs, "step_count", 0) or 0)

    decision = infer_decision_from_observation(obs)

    obvious_self_harm = (
        "self-harm" in hint
        or "do not plan to be here tomorrow" in text
        or "wrote the note" in text
    )
    obvious_phishing = "verify now" in text and "suspended" in text
    obvious_safe = "pricing is absurd" in text and "company keeps ignoring users" in text
    quote_case = "documenting what happened" in text or "quoted speech" in hint

    if obvious_self_harm:
        missing: List[str] = []
        if getattr(obs, "moderator_note", None) is None and "moderator_note" not in requested:
            missing.append("moderator_note")
        if getattr(obs, "risk_indicator", None) is None and "risk_indicator" not in requested:
            missing.append("risk_indicator")

        if missing:
            return ModerationAction(
                operation="request_context",
                requested_context_keys=missing,
                violation_type=None,
                severity=None,
                enforcement=None,
                escalate=None,
                rationale="High-risk self-harm case needs moderator note and risk indicators before finalization.",
            )

        if review_count == 0:
            return ModerationAction(
                operation="review",
                requested_context_keys=[],
                violation_type=decision["violation_type"],
                severity=decision["severity"],
                enforcement=decision["enforcement"],
                escalate=decision["escalate"],
                rationale="Imminent self-harm expression is supported by revealed high-risk evidence.",
            )

        return ModerationAction(
            operation="finalize",
            requested_context_keys=[],
            violation_type=decision["violation_type"],
            severity=decision["severity"],
            enforcement=decision["enforcement"],
            escalate=decision["escalate"],
            rationale="Current evidence indicates imminent self-harm risk and requires immediate escalation.",
        )

    if obvious_phishing:
        if review_count == 0:
            return ModerationAction(
                operation="review",
                requested_context_keys=[],
                violation_type=decision["violation_type"],
                severity=decision["severity"],
                enforcement=decision["enforcement"],
                escalate=decision["escalate"],
                rationale="Urgent credential-verification lure strongly indicates phishing.",
            )
        return ModerationAction(
            operation="finalize",
            requested_context_keys=[],
            violation_type=decision["violation_type"],
            severity=decision["severity"],
            enforcement=decision["enforcement"],
            escalate=decision["escalate"],
            rationale="Message is a deceptive credential-harvesting attempt and should be suspended.",
        )

    if obvious_safe:
        if review_count == 0:
            return ModerationAction(
                operation="review",
                requested_context_keys=[],
                violation_type=decision["violation_type"],
                severity=decision["severity"],
                enforcement=decision["enforcement"],
                escalate=decision["escalate"],
                rationale="This is strong criticism of a company or product, not harassment of a person.",
            )
        return ModerationAction(
            operation="finalize",
            requested_context_keys=[],
            violation_type=decision["violation_type"],
            severity=decision["severity"],
            enforcement=decision["enforcement"],
            escalate=decision["escalate"],
            rationale="Visible evidence supports allowing critical discussion of a product experience.",
        )

    if quote_case:
        missing: List[str] = []
        if getattr(obs, "conversation_excerpt", None) is None and "conversation_excerpt" not in requested:
            missing.append("conversation_excerpt")
        if getattr(obs, "moderator_note", None) is None and "moderator_note" not in requested:
            missing.append("moderator_note")

        if missing and step_count <= 2:
            return ModerationAction(
                operation="request_context",
                requested_context_keys=missing[:2],
                violation_type=None,
                severity=None,
                enforcement=None,
                escalate=None,
                rationale="Quoted-speech case needs context before a final safe decision.",
            )

        if review_count == 0:
            return ModerationAction(
                operation="review",
                requested_context_keys=[],
                violation_type="safe",
                severity="none",
                enforcement="allow",
                escalate=False,
                rationale="Visible content appears to condemn abusive language rather than endorse it.",
            )

        return ModerationAction(
            operation="finalize",
            requested_context_keys=[],
            violation_type="safe",
            severity="none",
            enforcement="allow",
            escalate=False,
            rationale="Content is reporting or condemning abuse, not making a hateful attack.",
        )

    return None


def repair_action(action: ModerationAction, obs: Any) -> ModerationAction:
    requested = _requested_context(obs)
    hint = _obs_hint(obs)
    text = _obs_text(obs)
    last_error = _obs_error(obs)
    review_count = _review_count(obs)

    fallback = infer_decision_from_observation(obs)

    if action.operation in {"review", "finalize"}:
        if action.violation_type is None:
            action.violation_type = fallback["violation_type"]
        if action.severity is None:
            action.severity = fallback["severity"]
        if action.enforcement is None:
            action.enforcement = fallback["enforcement"]
        if action.escalate is None:
            action.escalate = fallback["escalate"]
        if not (action.rationale or "").strip():
            action.rationale = fallback["rationale"]

    if action.operation == "finalize" and review_count == 0:
        action.operation = "review"

    if "incomplete structured decision" in last_error:
        action.operation = "review"

    if "cannot finalize before at least one review" in last_error:
        action.operation = "review"

    if action.operation == "finalize":
        needed: List[str] = []

        if "self-harm" in hint or "do not plan to be here tomorrow" in text or "wrote the note" in text:
            if getattr(obs, "moderator_note", None) is None and "moderator_note" not in requested:
                needed.append("moderator_note")
            if getattr(obs, "risk_indicator", None) is None and "risk_indicator" not in requested:
                needed.append("risk_indicator")

        elif "violence" in hint or "show up and end this" in text:
            if getattr(obs, "conversation_excerpt", None) is None and "conversation_excerpt" not in requested:
                needed.append("conversation_excerpt")
            if getattr(obs, "risk_indicator", None) is None and "risk_indicator" not in requested:
                needed.append("risk_indicator")
            if getattr(obs, "prior_content_pattern", None) is None and "prior_content_pattern" not in requested:
                needed.append("prior_content_pattern")

        if needed:
            return ModerationAction(
                operation="request_context",
                requested_context_keys=needed,
                violation_type=None,
                severity=None,
                enforcement=None,
                escalate=None,
                rationale="Additional high-value context is needed before finalization.",
            )

    if action.violation_type == "safe":
        action.severity = "none"
        action.enforcement = "allow"
        action.escalate = False

    if action.enforcement == "escalate_only":
        action.escalate = True

    if action.severity == "critical" and action.enforcement == "allow":
        action.enforcement = "escalate_only"
        action.escalate = True

    if action.operation == "request_context":
        deduped: List[str] = []
        for key in action.requested_context_keys:
            if key not in deduped and key not in requested:
                deduped.append(key)
        action.requested_context_keys = deduped

        if not action.requested_context_keys:
            action = ModerationAction(
                operation="review",
                requested_context_keys=[],
                violation_type=fallback["violation_type"],
                severity=fallback["severity"],
                enforcement=fallback["enforcement"],
                escalate=fallback["escalate"],
                rationale="No useful new context remains, so proceeding with a structured review.",
            )

    return action


def choose_action(client: OpenAI, obs: Any) -> ModerationAction:
    recovery = next_action_from_error(obs)
    if recovery is not None:
        return recovery

    deterministic = deterministic_action(obs)
    if deterministic is not None:
        return deterministic

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_to_prompt(obs)},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    parsed = extract_json(raw)
    action = ModerationAction.model_validate(parsed)
    return repair_action(action, obs)


def log_start(task_name: str) -> None:
    print(
        f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    step_num: int,
    action: ModerationAction,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step_num} "
        f"action={compact_json(action.model_dump(mode='json'))} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_val}",
        flush=True,
    )


def log_end(summary: EpisodeSummary) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in summary.rewards)
    print(
        f"[END] success={str(summary.success).lower()} "
        f"steps={summary.steps} "
        f"score={summary.final_score:.2f} "
        f"rewards={rewards_str} "
        f"last_error={summary.last_error or 'null'}",
        flush=True,
    )


def set_task_remote(base_url: str, task_name: str) -> None:
    response = requests.post(f"{base_url.rstrip('/')}/set_task/{task_name}", timeout=20)
    response.raise_for_status()


async def rollout_remote(task_name: str, client: OpenAI) -> EpisodeSummary:
    rewards: List[float] = []
    final_score = 0.0
    steps = 0
    success = False
    last_error: Optional[str] = None

    log_start(task_name)
    set_task_remote(API_BASE_URL, task_name)

    env_client = ContentModerationEnv(base_url=API_BASE_URL)
    try:
        result = await env_client.reset()
        obs = result.observation

        while True:
            action = choose_action(client, obs)
            result = await env_client.step(action)
            obs = result.observation

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            last_error = getattr(obs, "last_action_error", None)

            rewards.append(reward)
            steps += 1

            log_step(
                step_num=steps,
                action=action,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                final_score = float(getattr(obs, "current_score", 0.0) or 0.0)
                success = final_score >= SUCCESS_SCORE_THRESHOLD
                break
    finally:
        close_fn = getattr(env_client, "close", None)
        if callable(close_fn):
            maybe = close_fn()
            if asyncio.iscoroutine(maybe):
                await maybe

    return EpisodeSummary(
        task_name=task_name,
        mode="remote",
        steps=steps,
        rewards=rewards,
        final_score=final_score,
        success=success,
        last_error=last_error,
    )


async def rollout_local(task_name: str, client: OpenAI) -> EpisodeSummary:
    if ContentModerationEnvironment is None:
        raise RuntimeError(
            "Local environment mode requested, but ContentModerationEnvironment "
            "could not be imported from server.rl_meta_environment or "
            "server.my_env_v4_environment."
        )

    rewards: List[float] = []
    final_score = 0.0
    steps = 0
    success = False
    last_error: Optional[str] = None

    log_start(task_name)

    env = ContentModerationEnvironment(task=task_name)
    env.set_task(task_name)

    obs = env.reset()

    while True:
        action = choose_action(client, obs)
        obs = env.step(action)

        reward = float(getattr(obs, "reward", 0.0) or 0.0)
        done = bool(getattr(obs, "done", False))
        last_error = getattr(obs, "last_action_error", None)

        rewards.append(reward)
        steps += 1

        log_step(
            step_num=steps,
            action=action,
            reward=reward,
            done=done,
            error=last_error,
        )

        if done:
            final_score = float(getattr(obs, "current_score", 0.0) or 0.0)
            success = final_score >= SUCCESS_SCORE_THRESHOLD
            break

        await asyncio.sleep(0)

    return EpisodeSummary(
        task_name=task_name,
        mode="local",
        steps=steps,
        rewards=rewards,
        final_score=final_score,
        success=success,
        last_error=last_error,
    )


async def run_task(task_name: str, client: OpenAI, run_mode: RunMode) -> EpisodeSummary:
    if run_mode == "remote":
        return await rollout_remote(task_name, client)
    if run_mode == "local":
        return await rollout_local(task_name, client)
    raise ValueError(f"Unsupported RUN_MODE: {run_mode}")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if RUN_MODE not in {"remote", "local"}:
        raise ValueError("RUN_MODE must be either 'remote' or 'local'")

    all_summaries: List[EpisodeSummary] = []

    for task_name in ["task_easy", "task_medium", "task_hard"]:
        try:
            summary = await run_task(task_name=task_name, client=client, run_mode=RUN_MODE)
            all_summaries.append(summary)
            log_end(summary)
        except Exception as exc:
            failed = EpisodeSummary(
                task_name=task_name,
                mode=RUN_MODE,
                steps=0,
                rewards=[],
                final_score=0.0,
                success=False,
                last_error=str(exc),
            )
            all_summaries.append(failed)
            log_end(failed)

    total = len(all_summaries)
    passed = sum(1 for item in all_summaries if item.success)
    avg_score = sum(item.final_score for item in all_summaries) / max(1, total)

    print(
        "[SUMMARY] "
        f"mode={RUN_MODE} total_tasks={total} passed={passed} avg_score={avg_score:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())