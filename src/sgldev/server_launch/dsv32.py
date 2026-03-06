"""DeepSeek-V3.2 launch command."""

from sgldev.server_launch._base import make_launch_command

dsv32 = make_launch_command("deepseek-ai/DeepSeek-V3.2", tp=8)
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
