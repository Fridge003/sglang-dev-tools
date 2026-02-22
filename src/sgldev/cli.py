"""Main CLI entry point. Registers all sub-command groups."""

import typer

from sgldev.acc import app as acc_app
from sgldev.dev import app as dev_app
from sgldev.docker import app as docker_app
from sgldev.profile import app as profile_app
from sgldev.server import app as server_app
from sgldev.ssh import app as ssh_app

app = typer.Typer(
    name="sgldev",
    help="CLI toolkit for SGLang development, evaluation, profiling, and deployment.",
    no_args_is_help=True,
)

app.add_typer(acc_app, name="acc", help="Accuracy evaluation benchmarks")
app.add_typer(dev_app, name="dev", help="Development setup (git config, pre-commit, etc.)")
app.add_typer(server_app, name="server", help="SGLang server management")
app.add_typer(profile_app, name="profile", help="Profiling and benchmarking")
app.add_typer(docker_app, name="docker", help="Docker container management")
app.add_typer(ssh_app, name="ssh", help="SSH connection and rsync operations")

if __name__ == "__main__":
    app()
