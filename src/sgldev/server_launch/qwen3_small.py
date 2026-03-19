"""Qwen3-8B launch command."""

from sgldev.server_launch._base import make_launch_command

qwen3_8b = make_launch_command("Qwen/Qwen3-8B", tp=1)
qwen3_8b.__name__ = "qwen3_8b"
qwen3_8b.__doc__ = """\
Launch Qwen3-8B.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch qwen3-8b              # TP1 (default)
    sgldev server launch qwen3-8b --dp         # DP1
    sgldev server launch qwen3-8b --mtp        # TP1 + MTP
    sgldev server launch qwen3-8b --dp --mtp   # DP1 + MTP
"""
