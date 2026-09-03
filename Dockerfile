# Coding Agent — reproducible runtime image
#
# Build:
#   docker build -t coding-agent .
#
# Full stack (Postgres + Redis + API):
#   docker compose up --build
#
# Interactive chat on a host repo:
#   docker run --rm -it \
#     -e DEEPSEEK_API_KEY \
#     -v "$PWD:/workspace" \
#     coding-agent chat --repo /workspace

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENT_HOME=/opt/coding-agent

# git: GitTool / github_issue; curl: compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${AGENT_HOME}

COPY . .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[server]" \
    && pip install uv

RUN mkdir -p /workspace
WORKDIR /workspace

ENTRYPOINT ["agent"]
CMD ["--help"]
