"""Shared utilities: config, command execution, helpers."""

import os
import random
import subprocess
import time


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int = 0) -> int:
    return int(os.environ.get(key, str(default)))


def get_timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_random_id() -> int:
    return random.randint(0, 10000)


def log_tag() -> str:
    """Unique tag for log file naming: timestamp + random id."""
    return f"{get_timestamp()}_{get_random_id()}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def venv_cmd(venv_dir: str, cmd: str) -> str:
    """Wrap a command so it runs inside a venv."""
    return f"source {venv_dir}/bin/activate && {cmd}"


def run(cmd: str, capture_output: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Execute a shell command, printing it first."""
    print(f"EXEC: {cmd}", flush=True)
    return subprocess.run(
        ["bash", "-c", cmd],
        check=True,
        capture_output=capture_output,
        **({"text": True} if capture_output else {}),
        **kwargs,
    )


def build_env_string(env_vars: dict[str, str]) -> str:
    """Build 'KEY=VAL KEY2=VAL2 ...' string from a dict."""
    return " ".join(f"{k}={v}" for k, v in env_vars.items())


def build_flag_string(flags: dict[str, str | int | bool | None]) -> str:
    """Build CLI flag string from a dict.

    - bool True  -> --flag
    - bool False / None -> skipped
    - other      -> --flag value
    """
    parts: list[str] = []
    for key, val in flags.items():
        if val is None or val is False:
            continue
        flag = f"--{key}"
        if val is True:
            parts.append(flag)
        else:
            parts.append(f"{flag} {val}")
    return " ".join(parts)
