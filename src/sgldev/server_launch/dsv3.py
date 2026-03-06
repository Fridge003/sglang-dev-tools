"""DeepSeek-V3 launch command."""

from sgldev.server_launch._base import make_launch_command

dsv3 = make_launch_command("deepseek-ai/DeepSeek-V3-0324", tp=8)
dsv3.__name__ = "dsv3"
dsv3.__doc__ = """\
Launch DeepSeek-V3-0324.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch dsv3              # TP8 (default)
    sgldev server launch dsv3 --dp         # DP8
    sgldev server launch dsv3 --mtp        # TP8 + MTP
    sgldev server launch dsv3 --dp --mtp   # DP8 + MTP
"""
