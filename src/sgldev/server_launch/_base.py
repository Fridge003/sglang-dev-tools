"""Shared base configuration and helpers for server launch commands."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Annotated, Callable

import typer

from sgldev.common import log_tag, run
from sgldev.config import HOST, LOG_DIR, PORT


# ---------------------------------------------------------------------------
# Composable extra-argument strategies
#
# Each subclass represents a single feature flag (--dp, --mtp, …).
# To add a new flag:
#   1. Create an ExtraArgs subclass here.
#   2. Add a CLI parameter + ``if`` check in ``make_launch_command``.
# ---------------------------------------------------------------------------


class ExtraArgs(ABC):
    """A composable set of extra CLI arguments for server launch."""

    @abstractmethod
    def args(self) -> list[str]: ...


class DPArgs(ExtraArgs):
    """Data parallelism with DP attention."""

    def __init__(self, dp: int = 8):
        self.dp = dp

    def args(self) -> list[str]:
        return [f"--dp {self.dp}", "--enable-dp-attention"]


class MTPArgs(ExtraArgs):
    """MTP speculative decoding via EAGLE."""

    def args(self) -> list[str]:
        return [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
        ]


# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------


@dataclass
class LaunchConfig:
    """Server launch configuration composed from a model path and extras."""

    model_path: str = ""
    tp: int = 8
    host: str = HOST
    port: int = PORT
    extras: list[ExtraArgs] = field(default_factory=list)

    def build_cmd(self) -> str:
        parts = [
            "python3 -m sglang.launch_server",
            f"--model-path {self.model_path}",
            f"--tp {self.tp}",
        ]
        for extra in self.extras:
            parts.extend(extra.args())
        parts.extend([f"--host {self.host}", f"--port {self.port}"])
        return " ".join(parts)


def launch(config: LaunchConfig, background: bool, tee_log: bool) -> None:
    """Execute the server launch command."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    cmd = config.build_cmd()
    if tee_log:
        cmd += f" 2>&1 | tee {LOG_DIR}/server_{log_tag()}.log"
    elif background:
        cmd = f"nohup {cmd} > {LOG_DIR}/server_{log_tag()}.log 2>&1 &"
    run(cmd)


# ---------------------------------------------------------------------------
# Command factory
# ---------------------------------------------------------------------------


def make_launch_command(default_model_path: str) -> Callable:
    """Create a CLI launch command for a model.

    Each boolean flag (--dp, --mtp, …) maps to an ``ExtraArgs`` strategy.
    When a flag is enabled its extra arguments are appended to the command.
    """

    def command(
        dp: Annotated[
            bool, typer.Option("--dp", help="Enable DP=8 with DP attention")
        ] = False,
        mtp: Annotated[
            bool,
            typer.Option("--mtp", help="Enable MTP (EAGLE speculative decoding)"),
        ] = False,
        host: Annotated[str, typer.Option()] = HOST,
        port: Annotated[int, typer.Option()] = PORT,
        background: Annotated[
            bool, typer.Option(help="Run via nohup in background")
        ] = True,
        tee_log: Annotated[
            bool, typer.Option(help="Tee output to log file")
        ] = False,
    ) -> None:
        extras: list[ExtraArgs] = []
        if dp:
            extras.append(DPArgs())
        if mtp:
            extras.append(MTPArgs())

        config = LaunchConfig(
            model_path=default_model_path,
            extras=extras,
            host=host,
            port=port,
        )
        launch(config, background, tee_log)

    return command
