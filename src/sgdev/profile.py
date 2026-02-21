"""Profiling commands for SGLang (bench_one_batch_server, bench_serving)."""

from typing import Annotated

import typer

from sgdev.common import run
from sgdev.config import HOST, PORT

app = typer.Typer(no_args_is_help=True)


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def one_batch(
    ctx: typer.Context,
    model: Annotated[str, typer.Option(help="Model name (use 'none' for remote server)")] = "none",
    base_url: Annotated[str, typer.Option()] = "",
    batch_size: Annotated[int, typer.Option()] = 1,
    input_len: Annotated[int, typer.Option()] = 1024,
    output_len: Annotated[int, typer.Option()] = 10,
    skip_warmup: Annotated[bool, typer.Option()] = True,
    profile: Annotated[bool, typer.Option()] = True,
    profile_steps: Annotated[int, typer.Option()] = 10,
    profiler_dir: Annotated[str, typer.Option(help="SGLANG_TORCH_PROFILER_DIR")] = "",
):
    """Run sglang.bench_one_batch_server for single-batch latency profiling.

    Extra flags after `--` are forwarded verbatim.
    """
    url = base_url or f"http://{HOST}:{PORT}"

    env_prefix = ""
    if profiler_dir:
        env_prefix = f"SGLANG_TORCH_PROFILER_DIR={profiler_dir} "

    parts = [
        f"{env_prefix}python3 -m sglang.bench_one_batch_server",
        f"--model {model}",
        f"--base-url {url}",
        f"--batch-size {batch_size}",
        f"--input-len {input_len}",
        f"--output-len {output_len}",
    ]
    if skip_warmup:
        parts.append("--skip-warmup")
    if profile:
        parts.append("--profile")
    parts.append(f"--profile-steps {profile_steps}")

    if ctx.args:
        parts.extend(ctx.args)

    run(" ".join(parts))


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def serving(
    ctx: typer.Context,
    backend: Annotated[str, typer.Option()] = "sglang",
    num_prompts: Annotated[int, typer.Option()] = 64,
    dataset_name: Annotated[str, typer.Option()] = "random",
    random_input: Annotated[int, typer.Option()] = 1024,
    random_output: Annotated[int, typer.Option()] = 1024,
    random_range_ratio: Annotated[float, typer.Option()] = 1.0,
    max_concurrency: Annotated[int, typer.Option()] = 64,
    profile: Annotated[bool, typer.Option()] = True,
    host: Annotated[str, typer.Option()] = "",
    port: Annotated[int, typer.Option()] = 0,
):
    """Run sglang.bench_serving for throughput profiling.

    Extra flags after `--` are forwarded verbatim.
    """
    parts = [
        "python3 -m sglang.bench_serving",
        f"--backend {backend}",
        f"--num-prompts {num_prompts}",
        f"--dataset-name {dataset_name}",
        f"--random-input {random_input}",
        f"--random-output {random_output}",
        f"--random-range-ratio {random_range_ratio}",
        f"--max-concurrency {max_concurrency}",
    ]
    if host:
        parts.append(f"--host {host}")
    if port:
        parts.append(f"--port {port}")
    if profile:
        parts.append("--profile")

    if ctx.args:
        parts.extend(ctx.args)

    run(" ".join(parts))
