"""SGLang server launch helpers.

Provides pre-configured commands for launching DeepSeek-V3.2 in various
parallelism and speculative-decoding configurations.
"""

from dataclasses import dataclass
from typing import Annotated

import typer

from sgldev.common import log_tag, run
from sgldev.config import CUDA_VISIBLE_DEVICES, HOST, LOG_DIR, PORT

app = typer.Typer(no_args_is_help=True)


# ---------------------------------------------------------------------------
# DeepSeek-V3.2 launch configurations
# ---------------------------------------------------------------------------


@dataclass
class DeepSeekV32:
    """Base launch configuration for DeepSeek-V3.2."""

    model_path: str = "deepseek-ai/DeepSeek-V3.2"
    tp: int = 8
    host: str = HOST
    port: int = PORT

    def extra_args(self) -> list[str]:
        """Override in subclasses to add variant-specific arguments."""
        return []

    def build_cmd(self) -> str:
        parts = [
            f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}",
            "python3 -m sglang.launch_server",
            f"--model-path {self.model_path}",
            f"--tp {self.tp}",
        ]
        parts.extend(self.extra_args())
        parts.extend([f"--host {self.host}", f"--port {self.port}"])
        return " ".join(parts)


@dataclass
class DeepSeekV32TP8(DeepSeekV32):
    """TP8 — pure tensor parallelism."""


@dataclass
class DeepSeekV32DP8(DeepSeekV32):
    """DP8 — data parallelism with DP attention."""

    dp: int = 8

    def extra_args(self) -> list[str]:
        return [f"--dp {self.dp}", "--enable-dp-attention"]


@dataclass
class DeepSeekV32TP8MTP(DeepSeekV32):
    """TP8 + MTP — speculative decoding via EAGLE."""

    def extra_args(self) -> list[str]:
        return [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
        ]


@dataclass
class DeepSeekV32DP8MTP(DeepSeekV32DP8):
    """DP8 + MTP — data parallelism with speculative decoding."""

    def extra_args(self) -> list[str]:
        return [
            *super().extra_args(),
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _launch(config: DeepSeekV32, background: bool, tee_log: bool) -> None:
    cmd = config.build_cmd()
    if tee_log:
        cmd += f" 2>&1 | tee {LOG_DIR}/server_{log_tag()}.log"
    elif background:
        cmd = f"nohup {cmd} > {LOG_DIR}/server_{log_tag()}.log 2>&1 &"
    run(cmd)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

_CONFIGS = {
    (False, False): DeepSeekV32TP8,
    (True, False): DeepSeekV32DP8,
    (False, True): DeepSeekV32TP8MTP,
    (True, True): DeepSeekV32DP8MTP,
}


@app.command()
def dsv32(
    tp: Annotated[bool, typer.Option("--tp", help="Enable TP=8 (tensor parallelism)")] = True,
    dp: Annotated[bool, typer.Option("--dp", help="Enable DP=8 with DP attention")] = False,
    mtp: Annotated[bool, typer.Option("--mtp", help="Enable MTP (EAGLE speculative decoding)")] = False,
    host: Annotated[str, typer.Option()] = HOST,
    port: Annotated[int, typer.Option()] = PORT,
    background: Annotated[bool, typer.Option(help="Run via nohup in background")] = True,
    tee_log: Annotated[bool, typer.Option(help="Tee output to log file")] = False,
):
    """Launch DeepSeek-V3.2.

    Combine --tp, --dp, and --mtp to select a configuration:

        sgldev server dsv32              # TP8 (default)
        sgldev server dsv32 --dp         # DP8
        sgldev server dsv32 --mtp        # TP8 + MTP
        sgldev server dsv32 --dp --mtp   # DP8 + MTP
    """
    config_cls = _CONFIGS[(dp, mtp)]
    _launch(config_cls(host=host, port=port), background, tee_log)


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------


@app.command()
def health(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = PORT,
):
    """Check if the SGLang server is healthy."""
    run(f"curl -s http://{host}:{port}/health", capture_output=False)


@app.command()
def kill(
    port: Annotated[int, typer.Option()] = PORT,
):
    """Kill SGLang server process(es) listening on the given port."""
    run(f"fuser -k {port}/tcp || true")
