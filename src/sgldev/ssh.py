"""SSH and rsync helpers for remote machine access."""

from typing import Annotated

import typer

from sgldev.aliases import add as server_add
from sgldev.aliases import list_all as server_list_all
from sgldev.aliases import remove as server_remove
from sgldev.aliases import resolve as server_resolve
from sgldev.common import run

app = typer.Typer(no_args_is_help=True)


def _resolve_server(
    server: str
) -> tuple[str, str, str, int | None]:
    """Look up connection params for the given server name."""
    info = server_resolve(server)
    return info["user"], info["host"], info["key"], info.get("port", None)


def _ssh_base(user: str, host: str, key: str, port: int | None) -> list[str]:
    """Build the common ssh prefix: ssh -i key -p port user@host."""
    parts = ["ssh"]
    if key:
        parts.append(f"-i {key}")
    if port:
        parts.append(f"-p {port}")
    parts.append(f"{user}@{host}")
    return parts


# ── Connection commands ───────────────────────────────────────────────


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def connect(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name (defined via 'sgldev ssh server-set')")] = "",
    cmd: Annotated[str, typer.Option(help="Command to execute remotely (interactive shell if omitted)")] = "",
):
    """Open an interactive SSH session (or run a one-off remote command).
    sgldev ssh connect mybox
    sgldev ssh connect mybox --cmd "nvidia-smi"

    Extra flags after ``--`` are forwarded to ``ssh``, e.g.:

        sgldev ssh connect mybox -- -L 8080:localhost:8080
    """
    user, host, key, port = _resolve_server(server)
    parts = _ssh_base(user, host, key, port)

    extra_args = list(ctx.args) if ctx.args else []

    # --cmd may land in extra_args when placed after the positional server arg
    # (allow_interspersed_args=False stops option parsing after the positional)
    if "--cmd" in extra_args:
        idx = extra_args.index("--cmd")
        if idx + 1 < len(extra_args):
            cmd = extra_args[idx + 1]
            extra_args = extra_args[:idx] + extra_args[idx + 2:]

    if extra_args:
        parts.extend(extra_args)

    if cmd:
        parts.append(f'"{cmd}"')
    run(" ".join(parts))


@app.command()
def rsync(
    src: Annotated[str, typer.Argument(help="Source path (local or remote)")],
    dst: Annotated[str, typer.Argument(help="Destination path (local or remote)")],
    server: Annotated[str, typer.Option(help="Server name (defined via 'sgldev ssh server-set')")] = "",
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
        sgldev ssh rsync ./data /data --server mybox

        # pull from remote
        sgldev ssh rsync /data/results ./results --server mybox --no-to-remote
    """
    user, host, key, port = _resolve_server(server)

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


# ── Server management commands ────────────────────────────────────────


@app.command("server-set")
def server_set_cmd(
    name: Annotated[str, typer.Argument(help="Name for the server")],
    host: Annotated[str, typer.Option(help="Server IP / hostname")] = "",
    user: Annotated[str, typer.Option(help="SSH user (stored only if provided)")] = "",
    port: Annotated[int, typer.Option(help="SSH port (stored only if provided)")] = 0,
    key: Annotated[str, typer.Option(help="Path to SSH key (stored only if provided)")] = "",
):
    """Add or update a server.

    Examples::

        sgldev ssh server-set mybox --host 10.0.0.1
        sgldev ssh server-set mybox --host 10.0.0.1 --user root --port 2222
    """
    if not host:
        raise typer.BadParameter("--host is required when setting a server.")
    server_add(
        name,
        host=host,
        user=user or None,
        port=port or None,
        key=key or None,
    )
    print(f"Server '{name}' -> {host}")


@app.command("server-rm")
def server_rm_cmd(
    name: Annotated[str, typer.Argument(help="Server name to remove")],
):
    """Remove a server."""
    server_remove(name)
    print(f"Server '{name}' removed.")


@app.command("server-ls")
def server_ls_cmd():
    """List all servers."""
    data = server_list_all()
    if not data:
        print("No servers defined. Use 'sgldev ssh server-set <name> --host <ip>' to create one.")
        return
    max_name = max(len(e["server"]) for e in data)
    for entry in data:
        name = entry["server"]
        host = entry["host"]
        extras = []
        if "user" in entry:
            extras.append(f"user={entry['user']}")
        if "port" in entry:
            extras.append(f"port={entry['port']}")
        if "key" in entry:
            extras.append(f"key={entry['key']}")
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(f"  {name:<{max_name}}  {host}{suffix}")
