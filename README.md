# OpenEnv Content Moderation Benchmark

This repository contains a small content moderation benchmark built on OpenEnv.

An agent is shown a moderation case, may request limited hidden context, and must return a structured moderation decision. The environment scores both the decision quality and the workflow quality.

## What This Repo Contains

- `server/rl_meta_environment.py`: the benchmark logic and scoring rules
- `server/app.py`: a FastAPI wrapper that exposes the environment over HTTP
- `models.py`: shared typed models for actions, observations, and state
- `client.py`: typed OpenEnv client for talking to the server
- `inference.py`: the agent runner that uses an LLM to act inside the environment

## Benchmark Idea

The benchmark simulates a moderation queue with three task tiers:

- `task_easy`
- `task_medium`
- `task_hard`

At each step, the agent can choose one of three operations:

- `review`
- `request_context`
- `finalize`

The agent is not supposed to blindly finalize. It needs to behave like a reasonable moderator:

- request context only when it helps
- avoid over-enforcing safe content
- avoid under-enforcing dangerous content
- finalize only when the evidence is strong enough

## Brief Logic Overview

### `server/rl_meta_environment.py`

`ContentModerationEnvironment` is the core benchmark.

It does the following:

- stores a bank of synthetic moderation cases for each difficulty tier
- resets into one case at a time
- validates whether each action is structurally legal
- reveals requested hidden context when the request is valid
- rewards coherent intermediate behavior
- penalizes shortcuts, contradictions, weak evidence gathering, and invalid trajectories
- computes a final score when the agent finalizes a decision

The environment is intentionally designed so that the agent cannot score well by always doing the same thing. For example:

- `finalize` is blocked if no review has happened yet
- some ambiguous or critical cases require specific context before finalization
- repeated low-value behavior can be penalized
- structurally incomplete moderation decisions are rejected

In short, the environment evaluates both:

- what the final moderation decision is
- whether the agent followed a sensible moderation workflow

### `inference.py`

`inference.py` is the policy runner.

It loops through the benchmark tasks and, for each step:

1. reads the current observation from the environment
2. builds a prompt from visible content, policy snippets, revealed context, and workflow state
3. asks an LLM for one JSON moderation action
4. validates and repairs that action when needed
5. sends the action back to the environment
6. logs rewards, errors, and the final score

The file also includes deterministic helper logic before calling the model:

- rule-based recovery when the previous action failed
- heuristic handling for obvious phishing, self-harm, violence, or safe quote/reporting cases
- light repair logic to keep model outputs structurally valid

That means `inference.py` is not just a thin API call. It is a small agent policy with:

- prompting
- fallback heuristics
- action repair
- rollout orchestration

## Local vs Remote Mode

`inference.py` supports two run modes.

### Local mode

In local mode, the environment runs directly in Python.

Use this when you want the simplest setup:

```bash
export RUN_MODE=local
python inference.py
```

### Remote mode

In remote mode, the environment runs through the FastAPI server in `server/app.py`.

Start the server first:

```bash
uvicorn server.app:app --reload
```

Then run:

```bash
export RUN_MODE=remote
python inference.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt -r server/requirements.txt
```

## Environment Variables

`inference.py` currently reads:

- `HF_TOKEN`: required API key for the OpenAI-compatible client
- `API_BASE_URL`: base URL passed to the OpenAI client, and also used by remote mode for the environment server
- `MODEL_NAME`: model name sent to the LLM API
- `RUN_MODE`: `local` or `remote`

Current code defaults are:

- `RUN_MODE=remote`
- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`

Important note:

The current `inference.py` uses `API_BASE_URL` for both:

- the LLM endpoint
- the remote environment endpoint

So if you are using `remote` mode, make sure the code and your configuration are aligned with the service you want to call.

## Quick Start

### Option 1: simplest path with local mode

```bash
export RUN_MODE=local
export HF_TOKEN="your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Option 2: remote mode with FastAPI server

Terminal 1:

```bash
uvicorn server.app:app --reload
```

Terminal 2:

```bash
export RUN_MODE=remote
export HF_TOKEN="your_token_here"
export API_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

## API Endpoints

When the FastAPI server is running, these helper endpoints are available:

- `GET /health`
- `GET /tasks`
- `POST /set_task/{task_name}`

You also get the standard OpenEnv routes exposed by `create_fastapi_app(...)`.

## Typical Output

The runner prints progress like this:

```text
[START] task=task_easy env=openenv_content_moderation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"operation":"review",...} reward=0.05 done=false error=null
[STEP] step=2 action={"operation":"finalize",...} reward=0.16 done=true error=null
[END] success=true steps=2 score=0.81 rewards=0.05,0.16 last_error=null
[SUMMARY] mode=local total_tasks=3 passed=2 avg_score=0.7333
```

If you see `steps=0`, the run failed before the first environment step completed. Common causes are:

- invalid or missing `HF_TOKEN`
- unsupported `MODEL_NAME`
- wrong `API_BASE_URL`
- missing dependencies
- local import failures

## Suggested Reading Order

If you want to understand the code quickly, read in this order:

1. `models.py`
2. `server/rl_meta_environment.py`
3. `server/app.py`
4. `client.py`
5. `inference.py`
