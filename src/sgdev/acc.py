"""Accuracy evaluation benchmarks (GSM8K, GPQA, AIME, MMLU, LongBench)."""

import os
from typing import Annotated

import typer

from sgdev.common import (
    ensure_dir,
    env,
    env_int,
    get_random_id,
    log_tag,
    run,
    venv_cmd,
)

app = typer.Typer(no_args_is_help=True)

# ---------------------------------------------------------------------------
# Config from environment (with defaults suitable for container use)
# ---------------------------------------------------------------------------

MODEL_PATH = env("MODEL_PATH", "/data/weights/hello2026")
MODEL_PATH_NATIVE = env("MODEL_PATH_NATIVE", "/data/weights/hello2026_native")
CODE_PATH = env("CODE_PATH", "/sgl-workspace/NightFall")
CUDA_VISIBLE_DEVICES = env("CUDA_VISIBLE_DEVICES", "0,1,2,3")
HOST = env("HOST", "127.0.0.1")
PORT = env_int("PORT", 30010)
LOG_DIR = env("LOG_DIR", "/data/logs")

NS_VENV = env("NS_VENV", "/sgl-workspace/ns-venv")
LMEVAL_VENV = env("LMEVAL_VENV", "/sgl-workspace/lmeval-venv")
LONGBENCH_VENV = env("LONGBENCH_VENV", "/sgl-workspace/longbench-venv")


@app.callback()
def _banner():
    typer.echo(
        f"[acc] MODEL_PATH={MODEL_PATH}  HOST={HOST}:{PORT}  "
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}  LOG_DIR={LOG_DIR}"
    )


# ---- Server launching (kept here so `sgdev acc launch-server` still works) -


@app.command()
def launch_server(
    enable_all_optimization: Annotated[bool, typer.Option()] = False,
    enable_tp_attention: Annotated[bool, typer.Option()] = False,
    enable_thinking: Annotated[bool, typer.Option()] = True,
    v4_set_k_and_s_backend: Annotated[str, typer.Option()] = "triton",
    disable_cuda_graph: Annotated[bool, typer.Option()] = False,
    remove_gpu2cpu_sync: Annotated[bool, typer.Option()] = False,
    deepgemm_hc_prenorm: Annotated[bool, typer.Option()] = False,
    use_tilelang_mhc_pre: Annotated[bool, typer.Option()] = False,
    use_tilelang_mhc_post: Annotated[bool, typer.Option()] = False,
    use_triton_kcache_quant: Annotated[bool, typer.Option()] = False,
    use_fused_compress: Annotated[bool, typer.Option()] = False,
    pad_last_dim: Annotated[bool, typer.Option()] = False,
    enable_torch_compile: Annotated[bool, typer.Option()] = False,
    use_triton_rope: Annotated[bool, typer.Option()] = False,
    use_triton_rms_norm: Annotated[bool, typer.Option()] = False,
    use_split_k_mhc_pre: Annotated[bool, typer.Option()] = False,
    use_jit_kernel_fused_topk: Annotated[bool, typer.Option()] = False,
    use_tilelang_swa_prepare: Annotated[bool, typer.Option()] = False,
    use_multi_stream_overlap: Annotated[bool, typer.Option()] = False,
):
    """Launch the SGLang server with optimisation flags for accuracy testing."""
    os.chdir(CODE_PATH)
    log_dir = f"{LOG_DIR}/server_logs"
    ensure_dir(log_dir)

    on = enable_all_optimization

    def _flag(opt: bool) -> str:
        return "1" if opt or on else "0"

    env_vars = {
        "SGLANG_HACK_V4_SET_K_AND_S_BACKEND": v4_set_k_and_s_backend,
        "SGLANG_ENABLE_THINKING": "1" if enable_thinking else "0",
        "SGLANG_OPT_REMOVE_GPU2CPU_SYNC": _flag(remove_gpu2cpu_sync),
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": _flag(deepgemm_hc_prenorm),
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": _flag(use_tilelang_mhc_pre),
        "SGLANG_OPT_USE_TILELANG_MHC_POST": _flag(use_tilelang_mhc_post),
        "SGLANG_OPT_USE_TRITON_KCACHE_QUANT": _flag(use_triton_kcache_quant),
        "SGLANG_OPT_USE_FUSED_COMPRESS": _flag(use_fused_compress),
        "SGLANG_OPT_PAD_LAST_DIM": _flag(pad_last_dim),
        "SGLANG_OPT_DPSK_V4_ENABLE_TORCH_COMPILE": _flag(enable_torch_compile),
        "SGLANG_OPT_USE_TRITON_ROPE": _flag(use_triton_rope),
        "SGLANG_OPT_USE_TRITON_RMS_NORM": _flag(use_triton_rms_norm),
        "SGLANG_OPT_USE_SPLIT_K_MHC_PRE": _flag(use_split_k_mhc_pre),
        "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": _flag(use_jit_kernel_fused_topk),
        "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": _flag(use_tilelang_swa_prepare),
        "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": _flag(use_multi_stream_overlap),
    }

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())

    cmd = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} {env_str} "
    cmd += "SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 "
    cmd += "nohup python3 -m sglang.launch_server "
    cmd += f"--model-path {MODEL_PATH} --trust-remote-code --tp 4 "
    if not enable_tp_attention:
        cmd += "--dp 4 --enable-dp-attention "
    cmd += "--disable-radix-cache --attention-backend compressed "
    cmd += "--page-size 256 --max-running-request 64 "
    cmd += f"--chunked-prefill-size 8192 --watchdog-timeout 10000 "
    cmd += f"--host 0.0.0.0 --port {PORT} "
    if disable_cuda_graph:
        cmd += "--disable-cuda-graph "
    cmd += f"> {log_dir}/sglang_server_{log_tag()}.log 2>&1 &"

    run(cmd)


@app.command()
def launch_reference_server():
    """Launch the reference (torchrun) server for comparison."""
    os.chdir(os.path.join(CODE_PATH, "sunrise", "reference_implementation_updated"))
    log_dir = f"{LOG_DIR}/server_logs"
    ensure_dir(log_dir)

    run(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
        f"nohup torchrun --nproc-per-node 4 server.py "
        f"--ckpt-path {MODEL_PATH_NATIVE} "
        f"--config config_285B.json "
        f"--max-batch-size 8 "
        f"--port {PORT} "
        f"> {log_dir}/reference_server_{log_tag()}.log 2>&1 &"
    )


# ---- Setup commands -------------------------------------------------------


@app.command()
def setup_lmeval():
    """Set up lm-evaluation-harness in a dedicated venv."""
    run(f"uv venv {LMEVAL_VENV}")
    run("cd /sgl-workspace && git clone https://github.com/Fridge003/lm-evaluation-harness")
    run("cd /sgl-workspace/lm-evaluation-harness && git checkout gsm8k")
    run(venv_cmd(LMEVAL_VENV, "cd /sgl-workspace/lm-evaluation-harness && uv pip install -e '.[api]'"))


@app.command()
def setup_ns():
    """Set up NVIDIA NeMo Skills (ns) in a dedicated venv."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise typer.BadParameter("Set HF_TOKEN env var before running setup-ns")
    run(f"uv venv {NS_VENV}")
    run(venv_cmd(
        NS_VENV,
        "uv pip install git+https://github.com/NVIDIA-NeMo/Skills.git@d77caab --ignore-installed blinker",
    ))
    run(venv_cmd(NS_VENV, f"HF_TOKEN={hf_token} ns prepare_data mmlu aime24 aime25"))
    run(venv_cmd(NS_VENV, f"HF_TOKEN={hf_token} ns prepare_data gpqa --split diamond"))


@app.command()
def setup_longbench():
    """Set up LongBench in a dedicated venv."""
    run(f"uv venv {LONGBENCH_VENV}")
    run("cd /sgl-workspace && git clone https://github.com/Fridge003/LongBench.git")
    run(venv_cmd(LONGBENCH_VENV, "cd /sgl-workspace/LongBench && uv pip install -r requirements.txt"))


# ---- Evaluation commands ---------------------------------------------------


@app.command()
def run_gsm8k(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 1319,
    num_shots: Annotated[int, typer.Option()] = 5,
):
    """Run GSM8K evaluation (configurable few-shot)."""
    tag = log_tag()
    folder = f"{LOG_DIR}/gsm8k_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running GSM8K {num_shots}-shot  seed={seed}  log={folder}")

    run(venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k "
        f"--num_fewshot {num_shots} "
        f"--apply_chat_template "
        f"--limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {seed} --log_samples --output_path {folder} "
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_gsm8k_5_shots(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 100,
):
    """Run GSM8K CoT 5-shot evaluation."""
    tag = log_tag()
    folder = f"{LOG_DIR}/gsm8k_5_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running GSM8K CoT 5-shot  seed={seed}  log={folder}")

    run(venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k_cot_5shot "
        f"--num_fewshot 5 --apply_chat_template --limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {seed} --log_samples --output_path {folder} "
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_gsm8k_100_shots(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 100,
):
    """Run GSM8K CoT 100-shot evaluation."""
    tag = log_tag()
    folder = f"{LOG_DIR}/gsm8k_100_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running GSM8K CoT 100-shot  seed={seed}  log={folder}")

    run(venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k_cot_100shot "
        f"--num_fewshot 100 --apply_chat_template --limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {seed} --log_samples --output_path {folder} "
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_gpqa(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    """Run GPQA evaluation via NeMo Skills."""
    tag = log_tag()
    folder = f"{LOG_DIR}/gpqa_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running GPQA  repeats={num_repeats}  seed={seed}  log={folder}")

    run(venv_cmd(
        NS_VENV,
        f"nohup ns eval "
        f"--server_type=openai --model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=gpqa:{num_repeats} --output_dir={folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} ++inference.top_p=0.95 "
        f"++inference.timeout=25000000 --starting_seed {seed} "
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_mmlu(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
    max_samples: Annotated[int, typer.Option()] = -1,
):
    """Run MMLU evaluation via NeMo Skills."""
    tag = log_tag()
    folder = f"{LOG_DIR}/mmlu_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running MMLU  repeats={num_repeats}  seed={seed}  log={folder}")

    extra = f"++max_samples={max_samples} " if max_samples > 0 else ""
    run(venv_cmd(
        NS_VENV,
        f"nohup ns eval "
        f"--server_type=openai --model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=mmlu:{num_repeats} --output_dir={folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} ++inference.top_p=0.95 "
        f"++inference.timeout=25000000 --starting_seed {seed} "
        f"{extra}"
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_aime25(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    """Run AIME25 evaluation via NeMo Skills."""
    tag = log_tag()
    folder = f"{LOG_DIR}/aime25_logs/{tag}"
    ensure_dir(folder)
    seed = get_random_id()
    typer.echo(f"Running AIME25  repeats={num_repeats}  seed={seed}  log={folder}")

    run(venv_cmd(
        NS_VENV,
        f"nohup ns eval "
        f"--server_type=openai --model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=aime25:{num_repeats} --output_dir={folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} ++inference.top_p=0.95 "
        f"++inference.timeout=25000000 --starting_seed {seed} "
        f"> {folder}/output.log 2>&1 &",
    ))


@app.command()
def run_longbench(
    n_proc: Annotated[int, typer.Option()] = 128,
    num_prompts: Annotated[int, typer.Option()] = 128,
    thinking: Annotated[int, typer.Option()] = 1,
    skip_too_long_prompt: Annotated[int, typer.Option()] = 1,
    max_new_tokens: Annotated[int, typer.Option()] = 5000,
    max_len: Annotated[int, typer.Option()] = 60000,
    temperature: Annotated[float, typer.Option()] = 0.0,
):
    """Run LongBench evaluation."""
    tag = log_tag()
    folder = f"{LOG_DIR}/longbench_logs/{tag}"
    ensure_dir(folder)
    typer.echo(f"Running LongBench  log={folder}")

    run(venv_cmd(
        LONGBENCH_VENV,
        f"cd /sgl-workspace/LongBench && "
        f"HACK_THINKING={thinking} "
        f"HACK_SKIP_TOO_LONG_PROMPT={skip_too_long_prompt} "
        f"HACK_MAX_NEW_TOKENS={max_new_tokens} "
        f"HACK_MAX_LEN={max_len} "
        f"HACK_TEMPERATURE={temperature} "
        f"nohup python3 pred.py "
        f"--model {MODEL_PATH} --n_proc {n_proc} --num_prompts {num_prompts} "
        f"--port {PORT} --save_dir {folder} "
        f"> {folder}/output.log 2>&1 &",
    ))
