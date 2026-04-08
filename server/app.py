# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fastapi import HTTPException
from openenv.core.env_server import create_fastapi_app

try:
    from models import ModerationAction, ModerationObservation
    from server.rl_meta_environment import ContentModerationEnvironment
except ImportError:
    from ..models import ModerationAction, ModerationObservation
    from .rl_meta_environment import ContentModerationEnvironment


# IMPORTANT:
# Pass the environment CLASS, not an instance.
app = create_fastapi_app(
    ContentModerationEnvironment,
    ModerationAction,
    ModerationObservation,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "benchmark": "openenv_content_moderation",
    }


@app.get("/tasks")
def tasks() -> dict:
    # create a temporary env instance for this helper endpoint
    env = ContentModerationEnvironment()
    return {
        "benchmark": "openenv_content_moderation",
        "tasks": env.list_tasks(),
    }


@app.post("/set_task/{task_name}")
def set_task(task_name: str) -> dict:
    valid_tasks = {"task_easy", "task_medium", "task_hard"}
    if task_name not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_name}")

    # This does NOT affect OpenEnv per-session environments created by the server.
    # It only validates the task name and confirms availability.
    env = ContentModerationEnvironment()
    env.set_task(task_name)

    return {
        "status": "ok",
        "active_task": task_name,
        "message": "Task validated. Use reset with the desired task in your environment flow.",
    }