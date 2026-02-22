"""Development setup helpers: git config, pre-commit, etc."""

from typing import Annotated

import typer

from sgldev.common import run
from sgldev.config import GIT_EMAIL, GIT_NAME

app = typer.Typer(no_args_is_help=True)

@app.command("setup-sglang")
def setup_sglang(
    github_token: Annotated[str, typer.Argument(help="GitHub token")] = "",
    name: Annotated[str, typer.Option(help="Git user.name for this repo (default: $GIT_NAME)")] = GIT_NAME,
    email: Annotated[str, typer.Option(help="Git user.email for this repo (default: $GIT_EMAIL)")] = GIT_EMAIL,
):
    """Setup the sglang repository.

    Examples::

        sgldev dev setup-sglang <github_token>
    """
    if not github_token:
        raise typer.BadParameter("GitHub token is required.")

    run(f"cd /sgl-workspace && rm -rf sglang && git clone https://{github_token}@github.com/sgl-project/sglang.git")
    run(f"cd /sgl-workspace/sglang && pre-commit install")
    run(f"cd /sgl-workspace/sglang && pip install -e python")
    if not name and not email:
        raise typer.BadParameter("Provide at least one of --name or --email.")
    if name:
        run(f'cd /sgl-workspace/sglang && git config --local user.name "{name}"')
    if email:
        run(f'cd /sgl-workspace/sglang && git config --local user.email "{email}"')