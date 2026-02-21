"""Accuracy evaluation benchmarks (GSM8K, GPQA, AIME, MMLU, LongBench)."""

import os
from typing import Annotated

import typer

from sgldev.common import ensure_dir, get_random_id, log_tag, run, venv_cmd
from sgldev.config import (
    ACC_PORT as PORT,
    CUDA_VISIBLE_DEVICES,
    HOST,
    LMEVAL_VENV,
    LOG_DIR,
    LONGBENCH_VENV,
    MODEL_PATH,
    MODEL_PATH_NATIVE,
    NS_VENV,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _banner():
    typer.echo(
        f"[acc] MODEL_PATH={MODEL_PATH}  HOST={HOST}:{PORT}  "
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}  LOG_DIR={LOG_DIR}"
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
