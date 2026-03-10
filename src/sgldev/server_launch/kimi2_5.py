"""Kimi-2.5 launch command."""

from sgldev.server_launch._base import make_launch_command

kimi25 = make_launch_command("moonshotai/Kimi-K2.5", tp=8)
kimi25.__name__ = "kimi25"
kimi25.__doc__ = """\
Launch Kimi-K2.5.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch kimi25              # TP8 (default)
    sgldev server launch kimi25 --dp         # DP8
    sgldev server launch kimi25 --mtp        # TP8 + MTP
    sgldev server launch kimi25 --dp --mtp   # DP8 + MTP
"""
