"""GLM-5 launch configurations."""

from dataclasses import dataclass

from sgldev.server_launch._base import LaunchConfig, make_launch_command


@dataclass
class GLM5(LaunchConfig):
    """Base launch configuration for GLM-5."""

    model_path: str = "zai-org/GLM-5-FP8"


@dataclass
class GLM5TP8(GLM5):
    """TP8 — pure tensor parallelism."""


@dataclass
class GLM5DP8(GLM5):
    """DP8 — data parallelism with DP attention."""

    dp: int = 8

    def extra_args(self) -> list[str]:
        return [f"--dp {self.dp}", "--enable-dp-attention"]


@dataclass
class GLM5TP8MTP(GLM5):
    """TP8 + MTP — speculative decoding via EAGLE."""

    def extra_args(self) -> list[str]:
        return [
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 3",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 4",
        ]


@dataclass
class GLM5DP8MTP(GLM5DP8):
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
    (False, False): GLM5TP8,
    (True, False): GLM5DP8,
    (False, True): GLM5TP8MTP,
    (True, True): GLM5DP8MTP,
}

glm5 = make_launch_command(_CONFIGS)
glm5.__name__ = "glm5"
glm5.__doc__ = """\
Launch GLM-5.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch glm5              # TP8 (default)
    sgldev server launch glm5 --dp         # DP8
    sgldev server launch glm5 --mtp        # TP8 + MTP
    sgldev server launch glm5 --dp --mtp   # DP8 + MTP
"""
