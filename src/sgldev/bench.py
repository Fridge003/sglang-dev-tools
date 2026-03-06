"""Benchmarking commands (bench_serving, etc.)."""

from typing import Annotated

import typer

from sgldev.common import run

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(
    num_prompts: Annotated[int, typer.Option(help="Number of prompts to send.")] = 1,
    input_len: Annotated[int, typer.Option(help="Random input token length.")] = 1024,
    output_len: Annotated[int, typer.Option(help="Random output token length.")] = 16,
    max_concurrency: Annotated[int, typer.Option(help="Maximum concurrent requests.")] = 1,
):
    """Run bench_serving with random data against a local SGLang server."""
    run(
        "python3 -m sglang.bench_serving"
        " --backend sglang"
        f" --num-prompts {num_prompts}"
        " --dataset-name random"
        f" --random-input {input_len}"
        f" --random-output {output_len}"
        " --random-range-ratio 1.0"
        f" --max-concurrency {max_concurrency}"
    )
