"""Main CLI entry point. Registers all sub-command groups."""

import typer

from sgdev.acc import app as acc_app
from sgdev.docker import app as docker_app
from sgdev.profile import app as profile_app
from sgdev.server import app as server_app

app = typer.Typer(
    name="sgdev",
    help="CLI toolkit for SGLang development, evaluation, profiling, and deployment.",
    no_args_is_help=True,
)

app.add_typer(acc_app, name="acc", help="Accuracy evaluation benchmarks")
app.add_typer(server_app, name="server", help="SGLang server management")
app.add_typer(profile_app, name="profile", help="Profiling and benchmarking")
app.add_typer(docker_app, name="docker", help="Docker container management")

if __name__ == "__main__":
    app()
