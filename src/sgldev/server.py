"""SGLang server launch helpers.

Provides a flexible `launch` command that accepts common flags and forwards
any extra arguments directly to `sglang.launch_server`, so you never need to
modify this file just to pass a new flag.
"""

from typing import Annotated

import typer

from sgldev.common import log_tag, run
from sgldev.config import CUDA_VISIBLE_DEVICES, LOG_DIR, PORT
from sgldev.config import HOST

app = typer.Typer(no_args_is_help=True)


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def launch(
    ctx: typer.Context,
    model_path: Annotated[str, typer.Option(help="HF model id or local path")] = "deepseek-ai/DeepSeek-V3-0324",
    tp: Annotated[int, typer.Option(help="Tensor parallel size")] = 4,
    dp: Annotated[int, typer.Option(help="Data parallel size (0 = disabled)")] = 0,
    ep: Annotated[int, typer.Option(help="Expert parallel size (0 = disabled)")] = 0,
    enable_dp_attention: Annotated[bool, typer.Option()] = False,
    enable_dp_lm_head: Annotated[bool, typer.Option()] = False,
    kv_cache_dtype: Annotated[str, typer.Option()] = "",
    attention_backend: Annotated[str, typer.Option()] = "",
    moe_runner_backend: Annotated[str, typer.Option()] = "",
    moe_a2a_backend: Annotated[str, typer.Option()] = "",
    moe_dense_tp_size: Annotated[int, typer.Option()] = 0,
    quantization: Annotated[str, typer.Option()] = "",
    speculative_algorithm: Annotated[str, typer.Option()] = "",
    speculative_num_steps: Annotated[int, typer.Option()] = 0,
    speculative_eagle_topk: Annotated[int, typer.Option()] = 0,
    speculative_num_draft_tokens: Annotated[int, typer.Option()] = 0,
    max_running_requests: Annotated[int, typer.Option()] = 0,
    chunked_prefill_size: Annotated[int, typer.Option()] = 0,
    cuda_graph_max_bs: Annotated[int, typer.Option()] = 0,
    mem_fraction_static: Annotated[float, typer.Option()] = 0.0,
    context_length: Annotated[int, typer.Option()] = 0,
    disable_radix_cache: Annotated[bool, typer.Option()] = False,
    disable_cuda_graph: Annotated[bool, typer.Option()] = False,
    disable_shared_experts_fusion: Annotated[bool, typer.Option()] = False,
    trust_remote_code: Annotated[bool, typer.Option()] = True,
    host: Annotated[str, typer.Option()] = HOST,
    port: Annotated[int, typer.Option()] = PORT,
    env_vars: Annotated[str, typer.Option(help="Extra env vars, e.g. 'K1=V1 K2=V2'")] = "",
    background: Annotated[bool, typer.Option(help="Run via nohup in background")] = True,
    tee_log: Annotated[bool, typer.Option(help="Tee output to log file instead of nohup")] = False,
):
    """Launch sglang.launch_server with common options.

    Any unrecognised flags after `--` are forwarded verbatim, e.g.:

        sgldev server launch --model-path foo -- --tool-call-parser deepseekv32
    """
    parts: list[str] = []

    parts.append(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")
    if env_vars:
        parts.append(env_vars)

    parts.append("python3 -m sglang.launch_server")
    parts.append(f"--model-path {model_path}")
    parts.append(f"--tp {tp}")

    if dp:
        parts.append(f"--dp {dp}")
    if ep:
        parts.append(f"--ep {ep}")
    if enable_dp_attention:
        parts.append("--enable-dp-attention")
    if enable_dp_lm_head:
        parts.append("--enable-dp-lm-head")
    if kv_cache_dtype:
        parts.append(f"--kv-cache-dtype {kv_cache_dtype}")
    if attention_backend:
        parts.append(f"--attention-backend {attention_backend}")
    if moe_runner_backend:
        parts.append(f"--moe-runner-backend {moe_runner_backend}")
    if moe_a2a_backend:
        parts.append(f"--moe-a2a-backend {moe_a2a_backend}")
    if moe_dense_tp_size:
        parts.append(f"--moe-dense-tp-size {moe_dense_tp_size}")
    if quantization:
        parts.append(f"--quantization {quantization}")
    if speculative_algorithm:
        parts.append(f"--speculative-algorithm {speculative_algorithm}")
    if speculative_num_steps:
        parts.append(f"--speculative-num-steps {speculative_num_steps}")
    if speculative_eagle_topk:
        parts.append(f"--speculative-eagle-topk {speculative_eagle_topk}")
    if speculative_num_draft_tokens:
        parts.append(f"--speculative-num-draft-tokens {speculative_num_draft_tokens}")
    if max_running_requests:
        parts.append(f"--max-running-requests {max_running_requests}")
    if chunked_prefill_size:
        parts.append(f"--chunked-prefill-size {chunked_prefill_size}")
    if cuda_graph_max_bs:
        parts.append(f"--cuda-graph-max-bs {cuda_graph_max_bs}")
    if mem_fraction_static > 0:
        parts.append(f"--mem-fraction-static {mem_fraction_static}")
    if context_length:
        parts.append(f"--context-length {context_length}")
    if disable_radix_cache:
        parts.append("--disable-radix-cache")
    if disable_cuda_graph:
        parts.append("--disable-cuda-graph")
    if disable_shared_experts_fusion:
        parts.append("--disable-shared-experts-fusion")
    if trust_remote_code:
        parts.append("--trust-remote-code")

    parts.append(f"--host {host}")
    parts.append(f"--port {port}")

    if ctx.args:
        parts.extend(ctx.args)

    cmd = " ".join(parts)

    if tee_log:
        cmd += f" 2>&1 | tee {LOG_DIR}/server_{log_tag()}.log"
    elif background:
        cmd = f"nohup {cmd} > {LOG_DIR}/server_{log_tag()}.log 2>&1 &"

    run(cmd)


@app.command()
def health(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = PORT,
):
    """Check if the SGLang server is healthy."""
    run(f"curl -s http://{host}:{port}/health", capture_output=False)


@app.command()
def kill(
    port: Annotated[int, typer.Option()] = PORT,
):
    """Kill SGLang server process(es) listening on the given port."""
    run(f"fuser -k {port}/tcp || true")
