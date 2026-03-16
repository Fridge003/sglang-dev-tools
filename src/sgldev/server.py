"""SGLang server management commands.

Model-specific launch configurations live under ``server_launch/``.
"""

from typing import Annotated

import typer

from sgldev.common import run
from sgldev.config import HOST, PORT
from sgldev.server_launch import app as launch_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(launch_app, name="launch", help="Launch model servers")


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------


@app.command()
def health(
    host: Annotated[str, typer.Option()] = HOST,
    port: Annotated[int, typer.Option()] = PORT,
):
    """Check if the SGLang server is healthy."""
    run(f"curl -s http://{host}:{port}/health", capture_output=False)


@app.command()
def flush(
    host: Annotated[str, typer.Option()] = HOST,
    port: Annotated[int, typer.Option()] = PORT,
):
    """Flush all KV cache contents on the server."""
    run(f"curl -s http://{host}:{port}/flush_cache", capture_output=False)


@app.command()
def kill(
    port: Annotated[int, typer.Option()] = PORT,
):
    """Kill SGLang server process(es) listening on the given port."""
    run(f"kill -9 $(lsof -t -i:{port})")
