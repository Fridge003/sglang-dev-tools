"""DeepSeek-V3.2 launch configurations."""

from dataclasses import dataclass

from sgldev.server_launch._base import LaunchConfig, make_launch_command


@dataclass
class DeepSeekV32(LaunchConfig):
    """Base launch configuration for DeepSeek-V3.2."""

    model_path: str = "deepseek-ai/DeepSeek-V3.2"


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


_CONFIGS = {
    (False, False): DeepSeekV32TP8,
    (True, False): DeepSeekV32DP8,
    (False, True): DeepSeekV32TP8MTP,
    (True, True): DeepSeekV32DP8MTP,
}

dsv32 = make_launch_command(_CONFIGS)
dsv32.__name__ = "dsv32"
dsv32.__doc__ = """\
Launch DeepSeek-V3.2.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch dsv32              # TP8 (default)
    sgldev server launch dsv32 --dp         # DP8
    sgldev server launch dsv32 --mtp        # TP8 + MTP
    sgldev server launch dsv32 --dp --mtp   # DP8 + MTP
"""
