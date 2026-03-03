"""Shared base configuration and helpers for server launch commands."""

from dataclasses import dataclass
import os
from typing import Annotated, Callable

import typer

from sgldev.common import log_tag, run
from sgldev.config import HOST, LOG_DIR, PORT


@dataclass
class LaunchConfig:
    """Base launch configuration for SGLang models."""

    model_path: str = ""
    tp: int = 8
    host: str = HOST
    port: int = PORT

    def extra_args(self) -> list[str]:
        """Override in subclasses to add variant-specific arguments."""
        return []

    def build_cmd(self) -> str:
        parts = [
            "python3 -m sglang.launch_server",
            f"--model-path {self.model_path}",
            f"--tp {self.tp}",
        ]
        parts.extend(self.extra_args())
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


def make_launch_command(
    configs: dict[tuple[bool, bool], type[LaunchConfig]],
) -> Callable:
    """Create a standard CLI launch function for a model.

    The returned function accepts --dp, --mtp, --host, --port, --background,
    and --tee-log options and launches the appropriate configuration.
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
        config_cls = configs[(dp, mtp)]
        launch(config_cls(host=host, port=port), background, tee_log)

    return command
