"""SSH and rsync helpers for remote machine access."""

from typing import Annotated

import typer

from sgldev.common import run
from sgldev.config import (
    SSH_HOST,
    SSH_KEY,
    SSH_PORT,
    SSH_USER,
)

app = typer.Typer(no_args_is_help=True)


def _ssh_base(user: str, host: str, key: str, port: int) -> list[str]:
    """Build the common ssh prefix: ssh -i key -p port user@host."""
    parts = ["ssh"]
    if key:
        parts.append(f"-i {key}")
    if port != 22:
        parts.append(f"-p {port}")
    parts.append(f"{user}@{host}")
    return parts


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def connect(
    ctx: typer.Context,
    user: Annotated[str, typer.Option(help="Remote user")] = SSH_USER,
    host: Annotated[str, typer.Option(help="Remote host / IP")] = SSH_HOST,
    key: Annotated[str, typer.Option(help="Path to SSH private key")] = SSH_KEY,
    port: Annotated[int, typer.Option(help="SSH port")] = SSH_PORT,
    cmd: Annotated[str, typer.Option(help="Command to execute remotely (interactive shell if omitted)")] = "",
):
    """Open an interactive SSH session (or run a one-off remote command).

    Extra flags after ``--`` are forwarded to ``ssh``, e.g.:

        sgldev ssh connect -- -L 8080:localhost:8080
    """
    parts = _ssh_base(user, host, key, port)

    if ctx.args:
        parts.extend(ctx.args)

    if cmd:
        parts.append(f'"{cmd}"')

    run(" ".join(parts))


@app.command()
def rsync(
    src: Annotated[str, typer.Argument(help="Source path (local or remote)")],
    dst: Annotated[str, typer.Argument(help="Destination path (local or remote)")],
    user: Annotated[str, typer.Option(help="Remote user")] = SSH_USER,
    host: Annotated[str, typer.Option(help="Remote host / IP")] = SSH_HOST,
    key: Annotated[str, typer.Option(help="Path to SSH private key")] = SSH_KEY,
    port: Annotated[int, typer.Option(help="SSH port")] = SSH_PORT,
    to_remote: Annotated[bool, typer.Option(help="Copy local src → remote dst")] = True,
    delete: Annotated[bool, typer.Option(help="Delete extraneous files on receiver")] = False,
    dry_run: Annotated[bool, typer.Option(help="Show what would be transferred")] = False,
    exclude: Annotated[list[str], typer.Option(help="Patterns to exclude")] = [],
):
    """Rsync files between local and remote machines.

    By default copies *to* the remote (``--to-remote``).
    Pass ``--no-to-remote`` to pull *from* the remote instead.

    Examples::

        # push local dir to remote
        sgldev ssh rsync ./data /data --host 10.0.0.1

        # pull from remote
        sgldev ssh rsync /data/results ./results --host 10.0.0.1 --no-to-remote
    """
    ssh_cmd = "ssh"
    if key:
        ssh_cmd += f" -i {key}"
    if port != 22:
        ssh_cmd += f" -p {port}"

    parts = ["rsync", "-avz", "--progress"]
    parts.append(f'-e "{ssh_cmd}"')

    if delete:
        parts.append("--delete")
    if dry_run:
        parts.append("--dry-run")
    for pat in exclude:
        parts.append(f"--exclude '{pat}'")

    remote_prefix = f"{user}@{host}:"

    if to_remote:
        parts.append(src)
        parts.append(f"{remote_prefix}{dst}")
    else:
        parts.append(f"{remote_prefix}{src}")
        parts.append(dst)

    run(" ".join(parts))
