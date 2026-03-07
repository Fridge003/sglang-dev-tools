"""Server launch command group — one sub-command per model."""

import typer

from sgldev.server_launch.dsv32 import dsv32
from sgldev.server_launch.dsv32_fp4 import dsv32_fp4
from sgldev.server_launch.glm5 import glm5

app = typer.Typer(no_args_is_help=True)
app.command()(dsv32)
app.command()(dsv32_fp4)
app.command()(glm5)
