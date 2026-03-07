"""DeepSeek-V3.2 NVFP4 launch command."""

from sgldev.server_launch._base import make_launch_command

dsv32_fp4 = make_launch_command(
    "nvidia/DeepSeek-V3.2-NVFP4",
    tp=4,
    fixed_args=["--quantization", "modelopt_fp4"],
)
dsv32_fp4.__name__ = "dsv32_fp4"
dsv32_fp4.__doc__ = """\
Launch DeepSeek-V3.2 NVFP4.

Combine --dp and --mtp to select a configuration:

\b
    sgldev server launch dsv32_fp4              # TP4 (default)
    sgldev server launch dsv32_fp4 --dp         # DP4
    sgldev server launch dsv32_fp4 --mtp        # TP4 + MTP
    sgldev server launch dsv32_fp4 --dp --mtp   # DP4 + MTP
"""
