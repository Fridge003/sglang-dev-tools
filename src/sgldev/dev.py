"""Development setup helpers: git config, pre-commit, etc."""

from typing import Annotated

import typer

from sgldev.common import run
from sgldev.config import GIT_EMAIL, GIT_NAME

app = typer.Typer(no_args_is_help=True)


@app.command("git-config")
def git_config(
    name: Annotated[str, typer.Option(help="Git user.name for this repo (default: $GIT_NAME)")] = GIT_NAME,
    email: Annotated[str, typer.Option(help="Git user.email for this repo (default: $GIT_EMAIL)")] = GIT_EMAIL,
):
    """Set local git user.name and/or user.email.

    Examples::

        sgldev dev git-config --name "Alice" --email "alice@example.com"
        sgldev dev git-config --name "Alice"
    """
    if not name and not email:
        raise typer.BadParameter("Provide at least one of --name or --email.")
    if name:
        run(f'git config --local user.name "{name}"')
    if email:
        run(f'git config --local user.email "{email}"')


@app.command("refetch")
def refetch():
    """Re-add the sglang origin remote and fetch.

    Useful when the origin remote is stale or misconfigured.

    Examples::

        sgldev dev refetch
    """
    run("git remote rm origin")
    run("git remote add origin https://github.com/sgl-project/sglang.git")
    run("git fetch origin")


@app.command("pre-commit")
def pre_commit(
    run_all: Annotated[bool, typer.Option(help="Run pre-commit on all files after install")] = True,
):
    """Install pre-commit hooks and optionally run on all files.

    Examples::

        sgldev dev pre-commit
        sgldev dev pre-commit --no-run-all
    """
    run("pre-commit install")
    if run_all:
        run("pre-commit run --all-files")
