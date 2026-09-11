# The sandbox base image. Build once:
#
#     docker build -t heart-agent:latest .
#
# It carries what every role needs and no agent CLI at all. The CLIs are
# bind-mounted from the host at run time (see sandbox.AGENT_TOOLS), so adding an
# agent costs no bytes and no rebuild, and the container always runs the same
# build you have on the host. Docker's own agent template is 2.51GB and can run
# neither heart's api agent nor pytest; this is ~300MB and runs both.
#
# What it deliberately does NOT carry: a task's own test dependencies. No
# generic base can. Layer them on top per repo, or point HEART_SANDBOX_IMAGE at
# an image that has them:
#
#     FROM heart-agent:latest
#     COPY requirements.txt /tmp/
#     RUN pip install --no-cache-dir -r /tmp/requirements.txt
#
# Same for a toolchain this base does not carry at all: node's is here, but
# npm-test, cargo-test and go-test all want theirs installed.

# node 22, not Debian's 20: pi declares engines node>=22.19.0, and trixie ships
# 20.19.2. Copied as a single self-contained binary rather than installed --
# npm is not needed at run time because the CLIs' node_modules are mounted from
# the host too. Drop this stage and the COPY below to save ~120MB if you never
# run a node-based CLI (codex, pi) in the sandbox.
FROM node:22-slim AS node

FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node

# The verifiers detect.py emits for a Python repo: pytest, ruff, mypy. Without
# them a real repo's verify phase fails on "No module named pytest" and the
# episode scores `fail` -- the model billed for a missing dependency, which is
# the misattribution this whole feature exists to stop.
#
# detect.py decides whether to emit a ruff or mypy verifier by probing the
# *host*, so a host that has them and an image that does not is exactly how
# that failure arrives.
#
# The tools, not a task's own dependencies. Nothing generic can carry those --
# layer them on top and point HEART_SANDBOX_IMAGE at the result.
RUN pip install --no-cache-dir pytest ruff mypy

# heart itself, so `api:` agents can run: their command is
# `python3 -m heart.agents_api`, and without this it is a ModuleNotFoundError.
# heart declares no dependencies, so this adds essentially nothing.
COPY pyproject.toml /opt/heart/
COPY src /opt/heart/src
RUN pip install --no-cache-dir /opt/heart

# Mount points for the host's agent CLIs. They exist in the image so the binds
# land in a directory that is already there rather than one docker has to
# create inside a read-only rootfs.
#
# The agent user and the .docker/sandbox scaffolding are what the docker-sbx
# plugin requires of a template (HEART_SANDBOX=docker-sbx). Without them it
# refuses to start with "create lock file: ... No such file or directory".
# They cost nothing under HEART_SANDBOX=docker, so one image serves both.
RUN useradd -u 1000 -m -s /bin/bash agent \
 && mkdir -p /opt/agent-bin /opt/npm-global/bin \
             /home/agent/.docker/sandbox/locks /home/agent/workspace \
 && chown -R 1000:1000 /home/agent
ENV PATH=/opt/agent-bin:/opt/npm-global/bin:$PATH

# Matches the uid heart runs containers as by default (HEART_SANDBOX_USER
# overrides). Numeric, so no passwd entry is needed; HOME is set by the profile
# to a tmpfs path because the rootfs is read-only.
USER 1000:1000
