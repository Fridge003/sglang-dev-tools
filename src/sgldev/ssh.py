"""SSH and rsync helpers for remote machine access."""

from typing import Annotated

import typer

from sgldev.aliases import add as alias_add
from sgldev.aliases import list_all as alias_list_all
from sgldev.aliases import remove as alias_remove
from sgldev.aliases import resolve as alias_resolve
from sgldev.common import run

app = typer.Typer(no_args_is_help=True)


def _apply_alias(
    alias: str
) -> tuple[str, str, str, int | None]:
    """Override connection params with alias values when the CLI defaults weren't changed."""
    info = alias_resolve(alias)
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
    alias: Annotated[str, typer.Option(help="Server alias (defined via 'sgldev ssh alias-set')")] = "",
    cmd: Annotated[str, typer.Option(help="Command to execute remotely (interactive shell if omitted)")] = "",
):
    """Open an interactive SSH session (or run a one-off remote command).

    Extra flags after ``--`` are forwarded to ``ssh``, e.g.:

        sgldev ssh connect mybox -- -L 8080:localhost:8080
    """
    user, host, key, port = _apply_alias(alias)
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
    alias: Annotated[str, typer.Option(help="Server alias (defined via 'sgldev ssh alias-set')")] = "",
    to_remote: Annotated[bool, typer.Option(help="Copy local src → remote dst")] = True,
    delete: Annotated[bool, typer.Option(help="Delete extraneous files on receiver")] = False,
    dry_run: Annotated[bool, typer.Option(help="Show what would be transferred")] = False,
    exclude: Annotated[list[str], typer.Option(help="Patterns to exclude")] = [],
):
    """Rsync files between local and remote machines.

    By default copies *to* the remote (``--to-remote``).
    Pass ``--no-to-remote`` to pull *from* the remote instead.

    Examples::

        # push local dir to remote using alias
        sgldev ssh rsync ./data /data --alias mybox

        # pull from remote
        sgldev ssh rsync /data/results ./results --alias mybox --no-to-remote
    """
    user, host, key, port = _apply_alias(alias, user, host, key, port)

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


# ── Alias management commands ─────────────────────────────────────────


@app.command("alias-set")
def alias_set_cmd(
    name: Annotated[str, typer.Argument(help="Alias name for the server")],
    host: Annotated[str, typer.Option(help="Server IP / hostname")] = "",
    user: Annotated[str, typer.Option(help="SSH user (stored only if provided)")] = "",
    port: Annotated[int, typer.Option(help="SSH port (stored only if provided)")] = 0,
    key: Annotated[str, typer.Option(help="Path to SSH key (stored only if provided)")] = "",
):
    """Add or update a server alias.

    Examples::

        sgldev ssh alias-set mybox --host 10.0.0.1
        sgldev ssh alias-set mybox --host 10.0.0.1 --user root --port 2222
    """
    if not host:
        raise typer.BadParameter("--host is required when setting an alias.")
    alias_add(
        name,
        host=host,
        user=user or None,
        port=port or None,
        key=key or None,
    )
    print(f"Alias '{name}' -> {host}")


@app.command("alias-rm")
def alias_rm_cmd(
    name: Annotated[str, typer.Argument(help="Alias name to remove")],
):
    """Remove a server alias."""
    alias_remove(name)
    print(f"Alias '{name}' removed.")


@app.command("alias-ls")
def alias_ls_cmd():
    """List all server aliases."""
    data = alias_list_all()
    if not data:
        print("No aliases defined. Use 'sgldev ssh alias-set <name> --host <ip>' to create one.")
        return
    max_name = max(len(e["alias"]) for e in data)
    for entry in data:
        name = entry["alias"]
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
