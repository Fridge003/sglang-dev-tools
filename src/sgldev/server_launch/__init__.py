"""Server launch command group — one sub-command per model."""

import typer

from sgldev.server_launch.dsv32 import dsv32
from sgldev.server_launch.dsv32_fp4 import dsv32_fp4
from sgldev.server_launch.glm5 import glm5
from sgldev.server_launch.glm5_fp4 import glm5_fp4
from sgldev.server_launch.kimi2_5 import kimi25
from sgldev.server_launch.qwen3_30b import qwen3_30b
from sgldev.server_launch.qwen3_small import qwen3_8b

app = typer.Typer(no_args_is_help=True)
app.command()(dsv32)
app.command()(dsv32_fp4)
app.command()(glm5)
app.command()(glm5_fp4)
app.command()(kimi25)
app.command()(qwen3_8b)
app.command()(qwen3_30b)
