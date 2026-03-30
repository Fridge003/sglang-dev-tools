"""Docker container management for SGLang development."""

from typing import Annotated

import typer

from pathlib import Path

from sgldev.common import env, run
from sgldev.config import DEFAULT_CACHE, DEFAULT_CONTAINER, DEFAULT_IMAGE, DEFAULT_SHM, DEFAULT_SGLANG_PATH

app = typer.Typer(no_args_is_help=True)

DEFAULT_SHELL = "/bin/zsh"


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def create(
    ctx: typer.Context,
    name: Annotated[str, typer.Option(help="Container name")] = DEFAULT_CONTAINER,
    image: Annotated[str, typer.Option(help="Docker image")] = DEFAULT_IMAGE,
    cache_path: Annotated[str, typer.Option(help="Host path to mount as /root/.cache")] = DEFAULT_CACHE,
    hf_token: Annotated[str, typer.Option(help="HuggingFace token (or set HF_TOKEN env)")] = "",
    shm_size: Annotated[str, typer.Option(help="Shared memory size")] = DEFAULT_SHM,
    gpus: Annotated[str, typer.Option(help="GPU specification")] = "all",
    shell: Annotated[str, typer.Option(help="Shell to run")] = DEFAULT_SHELL,
    volumes: Annotated[list[str], typer.Option("-v", help="Extra volume mounts")] = [],
    envs: Annotated[list[str], typer.Option("-e", help="Extra env vars (K=V)")] = [],
    detach: Annotated[bool, typer.Option(help="Run in detached mode")] = True,
    ptrace: Annotated[bool, typer.Option(help="Add SYS_PTRACE capability")] = False,
):
    """Create and start a new SGLang development container.

    Extra flags after `--` are forwarded to `docker run`.

    Example:
        sgldev docker create --name mydev --cache-path /data/hf-cache --hf-token <huggingface_token>
    """
    Path(DEFAULT_SGLANG_PATH).mkdir(parents=True, exist_ok=True)

    parts = ["docker run"]
    if detach:
        parts.append("-itd")
    else:
        parts.append("-it")

    parts.append(f"--shm-size {shm_size}")
    parts.append(f"--gpus {gpus}")
    parts.append("--ipc=host")
    if ptrace:
        parts.append("--cap-add SYS_PTRACE")
    parts.append(f"--name {name}")

    if cache_path:
        parts.append(f'-v "{cache_path}":/root/.cache')

    parts.append(f'-v "{DEFAULT_SGLANG_PATH}":/sgl-workspace/sglang')

    for v in volumes:
        parts.append(f'-v "{v}"')

    token = hf_token or env("HF_TOKEN", "")
    if token:
        parts.append(f'--env "HF_TOKEN={token}"')

    for e in envs:
        parts.append(f'--env "{e}"')

    if ctx.args:
        parts.extend(ctx.args)

    parts.append(image)
    parts.append(shell)

    run(" ".join(parts))


@app.command(name="exec")
def exec_into(
    name: Annotated[str, typer.Option(help="Container name")] = DEFAULT_CONTAINER,
    shell: Annotated[str, typer.Option()] = DEFAULT_SHELL,
):
    """Exec into a running container."""
    run(f"docker exec -it {name} {shell}")

@app.command(name="pull")
def exec_pull(
    name: Annotated[str, typer.Option(help="Container name")] = DEFAULT_IMAGE,
):
    """Pull the latest image from the registry."""
    run(f"docker pull {name}")


@app.command()
def rm(
    name: Annotated[str, typer.Option(help="Container name")] = DEFAULT_CONTAINER,
    force: Annotated[bool, typer.Option()] = True,
):
    """Remove a container."""
    flag = " -f" if force else ""
    run(f"docker rm{flag} {name}")


@app.command()
def logs(
    name: Annotated[str, typer.Argument(help="Container name")],
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
    tail: Annotated[int, typer.Option()] = 100,
):
    """Show container logs."""
    follow_flag = " -f" if follow else ""
    run(f"docker logs{follow_flag} --tail {tail} {name}")


@app.command()
def killer():
    """Launch a privileged container with host PID namespace."""
    run("docker run --rm -it --privileged --pid=host ubuntu")


@app.command(name="list")
def list_containers(
    all_containers: Annotated[bool, typer.Option("--all", "-a")] = False,
    filter_name: Annotated[str, typer.Option(help="Filter by name substring")] = "sglang",
):
    """List containers, optionally filtered by name."""
    flag = " -a" if all_containers else ""
    filt = f' --filter "name={filter_name}"' if filter_name else ""
    run(f"docker ps{flag}{filt}")
