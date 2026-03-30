"""Shared base configuration and helpers for server launch commands."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
    fixed_args: list[str] = field(default_factory=list)
    extras: list[ExtraArgs] = field(default_factory=list)

    def build_cmd(self) -> str:
        parts = [
            "sglang serve",
            f"--model-path {self.model_path}",
            f"--tp {self.tp}",
            "--trust-remote-code",
        ]
        parts.extend(self.fixed_args)
        for extra in self.extras:
            parts.extend(extra.args())
        parts.extend([f"--host {self.host}", f"--port {self.port}"])
        return " ".join(parts)


def launch(config: LaunchConfig, tee_log: bool, prefix: str = "") -> None:
    """Execute the server launch command."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    cmd = config.build_cmd()
    if tee_log:
        cmd += f" 2>&1 | tee {LOG_DIR}/{prefix}_{log_tag()}.log"
    else:
        cmd = f"nohup {cmd} > {LOG_DIR}/{prefix}_{log_tag()}.log 2>&1 &"
    run(cmd)


# ---------------------------------------------------------------------------
# Command factory
# ---------------------------------------------------------------------------


def make_launch_command(
    default_model_path: str, tp: int = 8, fixed_args: list[str] | None = None
) -> Callable:
    """Create a CLI launch command for a model.

    Each boolean flag (--dp, --mtp, …) maps to an ``ExtraArgs`` strategy.
    When a flag is enabled its extra arguments are appended to the command.
    """

    fixed_args = fixed_args or []

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
        tee_log: Annotated[
            bool, typer.Option(help="Tee output to log file")
        ] = False,
    ) -> None:
        extras: list[ExtraArgs] = []
        if dp:
            extras.append(DPArgs(dp=tp))
        if mtp:
            extras.append(MTPArgs())

        config = LaunchConfig(
            model_path=default_model_path,
            tp=tp,
            fixed_args=fixed_args,
            extras=extras,
            host=host,
            port=port,
        )
        launch(config, tee_log, prefix=f"{default_model_path.split('/')[-1]}_TP{tp}_DP{dp}_MTP{mtp}")

    return command
