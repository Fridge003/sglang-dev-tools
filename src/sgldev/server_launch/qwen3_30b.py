"""Qwen3-30B-A3B launch command."""

from sgldev.server_launch._base import make_launch_command

qwen3_30b = make_launch_command("Qwen/Qwen3-30B-A3B", tp=1)
qwen3_30b.__name__ = "qwen3_30b"
qwen3_30b.__doc__ = """\
Launch Qwen3-30B-A3B.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch qwen3-30b              # TP1 (default)
    sgldev server launch qwen3-30b --dp         # DP1
    sgldev server launch qwen3-30b --mtp        # TP1 + MTP
    sgldev server launch qwen3-30b --dp --mtp   # DP1 + MTP
"""
