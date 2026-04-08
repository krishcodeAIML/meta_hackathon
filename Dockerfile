# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env
WORKDIR /app/env

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN if [ -f requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r requirements.txt; \
    elif [ -f server/requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r server/requirements.txt; \
    else \
        echo "No requirements.txt found"; exit 1; \
    fi

RUN python3 -c "import openenv; print('openenv ok:', openenv.__file__)"
RUN python3 -c "from openenv.core.env_server import Environment; print('Environment import ok')"

ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]