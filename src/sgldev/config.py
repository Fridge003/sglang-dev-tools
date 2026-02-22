"""Centralized environment variable definitions."""

from sgldev.common import env, env_int

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES = env("CUDA_VISIBLE_DEVICES", "0,1,2,3")
LOG_DIR = env("LOG_DIR", "/data/logs")
HOST = env("HOST", "127.0.0.1")
PORT = env_int("PORT", 30000)

# ---------------------------------------------------------------------------
# Accuracy benchmarks (acc.py)
# ---------------------------------------------------------------------------
MODEL_PATH = env("MODEL_PATH", "/data/weights/hello2026")
MODEL_PATH_NATIVE = env("MODEL_PATH_NATIVE", "/data/weights/hello2026_native")
CODE_PATH = env("CODE_PATH", "/sgl-workspace/NightFall")
ACC_PORT = env_int("PORT", 30010)
NS_VENV = env("NS_VENV", "/sgl-workspace/ns-venv")
LMEVAL_VENV = env("LMEVAL_VENV", "/sgl-workspace/lmeval-venv")
LONGBENCH_VENV = env("LONGBENCH_VENV", "/sgl-workspace/longbench-venv")

# ---------------------------------------------------------------------------
# Docker (docker.py)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE = env("SGDEV_DOCKER_IMAGE", "lmsysorg/sglang:dev")
DEFAULT_SHM = env("SGDEV_DOCKER_SHM", "32g")
DEFAULT_CACHE = env("SGDEV_DOCKER_CACHE", "")
DEFAULT_CONTAINER = env("SGDEV_DOCKER_CONTAINER", "sglang_baizhou")

# ---------------------------------------------------------------------------
# Server (server.py) — listens on 0.0.0.0 by default
# ---------------------------------------------------------------------------
SERVER_HOST = env("HOST", "0.0.0.0")
