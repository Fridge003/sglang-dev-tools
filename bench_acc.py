# The latest script can be fetched from https://github.com/DarkSharpness/NightFall/blob/script/bench.py
# This script should be used inside the container. Before testing anything, please
# 1. install typer
# 2. set the following environment variables:
# - MODEL_PATH: the path to the model
# - CODE_PATH: the path to the code
# - CUDA_VISIBLE_DEVICES: the visible devices
# - PORT: the port to listen on
# - LOG_DIR: the path to the logs
# 3. checkout to the commit you want to test

# Caution!!!
# This script assumes that thinking mode can be controlled from SGLang side. (with an environ or argument)
# e.g. ++chat_template_kwargs.thinking=true is not included in the nemo skills command

# Launch server:
#    DP4:
#        CUDA_VISIBLE_DEVICES=0,1,2,3 python bench_acc.py launch-server --enable-thinking --enable-all-optimization (or appending other arguments when needed)
#    TP4:
#        CUDA_VISIBLE_DEVICES=4,5,6,7 python bench_acc.py launch-server --enable-thinking --enable-all-optimization --enable-tp-attention (or appending other arguments when needed)

# Launch reference server:
#    CUDA_VISIBLE_DEVICES=4,5,6,7 python bench_acc.py launch-reference-server

# Test gsm8k (without cot):
#    python bench_acc.py setup-lmeval
#    python bench_acc.py run-gsm8k --temperature 0.0 --max-tokens 50000 --max-samples 1319 --num-shots 5

# Test gsm8k cot 5 shots:
#    python bench_acc.py setup-lmeval
#    python bench_acc.py run-gsm8k-5-shots --temperature 0.0 --max-tokens 50000 --max-samples 1319

# Test gsm8k cot 100 shots:
#    python bench_acc.py setup-lmeval
#    python bench_acc.py run-gsm8k-100-shots --temperature 0.0 --max-tokens 50000 --max-samples 1319

# Test GPQA:
#    python bench_acc.py setup-ns
#    python bench_acc.py run-gpqa --num-repeats 16 --temperature 1.0 --max-tokens 60000 --max-concurrency 64

# Test AIME25:
#    python bench_acc.py setup-ns
#    python bench_acc.py run-aime25 --num-repeats 16 --temperature 1.0 --max-tokens 60000 --max-concurrency 64

# Test MMLU:
#    python bench_acc.py setup-ns
#    python bench_acc.py run-mmlu --num-repeats 16 --temperature 1.0 --max-tokens 60000 --max-concurrency 64 --max-samples 500

# Test longbench:
#    python bench_acc.py setup-longbench
#    python bench_acc.py run-longbench --thinking 1 --skip-too-long-prompt 1 --max-new-tokens 5000 --max-len 60000 --temperature 0


import os
import random
import subprocess
import time
from typing import Annotated

import typer

app = typer.Typer()

# Some manually set configs:
MODEL_PATH = "/data/weights/hello2026"
MODEL_PATH_NATIVE = "/data/weights/hello2026_native"
CODE_PATH = os.environ.get("CODE_PATH", "/sgl-workspace/NightFall")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "30010"))
LOG_DIR = "/data/logs"

NS_VENV = "/sgl-workspace/ns-venv"
LMEVAL_VENV = "/sgl-workspace/lmeval-venv"
LONGBENCH_VENV = "/sgl-workspace/longbench-venv"

info_msg = f"""
Using configurations:
MODEL_PATH: {MODEL_PATH}
CODE_PATH: {CODE_PATH}
CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}
HOST: {HOST}
PORT: {PORT}
LOG_DIR: {LOG_DIR}
"""

# input(info_msg + "\nPress Enter to continue...")
print(info_msg)


def _venv_cmd(venv_dir: str, cmd: str) -> str:
    return f"source {venv_dir}/bin/activate && {cmd}"


def get_timestamp():
    return time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))


def get_random_int():
    return random.randint(0, 10000)


# @app.command()
# def setup_sglang_env(commit_hash=None):
#     exec_command(f"mkdir -p {LOG_DIR}")

#     # Clone NightFall repo
#     os.chdir("/sgl-workspace")
#     exec_command(f"git clone https://{GITHUB_TOKEN}@github.com/DarkSharpness/NightFall.git")
#     os.chdir("/sgl-workspace/NightFall")
#     if commit_hash is not None:
#         exec_command(f"git checkout {commit_hash}")
#     else:
#         exec_command(f"git checkout origin/final")
#     exec_command(f"pip install -e 'python'")

#     # Install tilelang/flashmla
#     os.chdir("/sgl-workspace")
#     exec_command(f"pip install tilelang")
#     exec_command(f"git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla")
#     os.chdir("flash-mla")
#     exec_command(f"git submodule update --init --recursive")
#     exec_command(f"pip install --no-build-isolation -v .")


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
    os.chdir(CODE_PATH)
    if not os.path.exists(f"{LOG_DIR}/server_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/server_logs")

    open_all = enable_all_optimization
    env_vars = {
        "SGLANG_HACK_V4_SET_K_AND_S_BACKEND": v4_set_k_and_s_backend,
        "SGLANG_ENABLE_THINKING": "1" if enable_thinking else "0",
        "SGLANG_OPT_REMOVE_GPU2CPU_SYNC": (
            "1" if remove_gpu2cpu_sync or open_all else "0"
        ),
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": (
            "1" if deepgemm_hc_prenorm or open_all else "0"
        ),
        "SGLANG_OPT_USE_TILELANG_MHC_PRE": (
            "1" if use_tilelang_mhc_pre or open_all else "0"
        ),
        "SGLANG_OPT_USE_TILELANG_MHC_POST": (
            "1" if use_tilelang_mhc_post or open_all else "0"
        ),
        "SGLANG_OPT_USE_TRITON_KCACHE_QUANT": (
            "1" if use_triton_kcache_quant or open_all else "0"
        ),
        "SGLANG_OPT_USE_FUSED_COMPRESS": "1" if use_fused_compress or open_all else "0",
        "SGLANG_OPT_PAD_LAST_DIM": "1" if pad_last_dim or open_all else "0",
        "SGLANG_OPT_DPSK_V4_ENABLE_TORCH_COMPILE": (
            "1" if enable_torch_compile or open_all else "0"
        ),
        "SGLANG_OPT_USE_TRITON_ROPE": "1" if use_triton_rope or open_all else "0",
        "SGLANG_OPT_USE_TRITON_RMS_NORM": (
            "1" if use_triton_rms_norm or open_all else "0"
        ),
        "SGLANG_OPT_USE_SPLIT_K_MHC_PRE": (
            "1" if use_split_k_mhc_pre or open_all else "0"
        ),
        "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": (
            "1" if use_jit_kernel_fused_topk or open_all else "0"
        ),
        "SGLANG_OPT_USE_TILELANG_SWA_PREPARE": (
            "1" if use_tilelang_swa_prepare or open_all else "0"
        ),
        "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": (
            "1" if use_multi_stream_overlap or open_all else "0"
        ),
    }

    for key, value in env_vars.items():
        print(f"{key}={value}")

    launch_command = ""
    launch_command += f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
    for key, value in env_vars.items():
        launch_command += f"{key}={value} "
    launch_command += f"SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 "
    launch_command += f"nohup python3 -m sglang.launch_server "
    launch_command += f"--model-path {MODEL_PATH} "
    launch_command += f"--trust-remote-code "
    launch_command += f"--tp 4 "
    if not enable_tp_attention:
        launch_command += f"--dp 4 "
        launch_command += f"--enable-dp-attention "
    launch_command += f"--disable-radix-cache "
    launch_command += f"--attention-backend compressed "
    launch_command += f"--page-size 256 "
    launch_command += f"--max-running-request 64 "
    launch_command += f"--chunked-prefill-size 8192 "
    launch_command += f"--watchdog-timeout 10000 "
    launch_command += f"--host 0.0.0.0 "
    launch_command += f"--port {PORT} "

    if disable_cuda_graph:
        launch_command += f"--disable-cuda-graph "

    launch_command += f"> {LOG_DIR}/server_logs/sglang_server_{get_timestamp()}_{get_random_int()}.log 2>&1 &"
    exec_command(launch_command)


@app.command()
def launch_reference_server():
    os.chdir(os.path.join(CODE_PATH, "sunrise", "reference_implementation_updated"))
    if not os.path.exists(f"{LOG_DIR}/server_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/server_logs")

    exec_command(
        f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
        f"nohup torchrun --nproc-per-node 4 server.py "
        f"--ckpt-path {MODEL_PATH_NATIVE} "
        f"--config config_285B.json "
        f"--max-batch-size 8 "
        f"--port {PORT} "
        f"> {LOG_DIR}/server_logs/reference_server_{get_timestamp()}_{get_random_int()}.log 2>&1 &"
    )


@app.command()
def setup_lmeval():
    exec_command(f"uv venv {LMEVAL_VENV}")
    exec_command(
        f"cd /sgl-workspace && git clone https://github.com/Fridge003/lm-evaluation-harness"
    )
    exec_command(
        f"cd /sgl-workspace/lm-evaluation-harness && git checkout gsm8k"
    )
    exec_command(
        _venv_cmd(LMEVAL_VENV, "cd /sgl-workspace/lm-evaluation-harness && uv pip install -e '.[api]'")
    )


@app.command()
def run_gsm8k(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 1319,
    num_shots: Annotated[int, typer.Option()] = 5,
):
    if not os.path.exists(f"{LOG_DIR}/gsm8k_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/gsm8k_logs")
    random_seed = get_random_int()
    gsm8k_log_folder = f"{LOG_DIR}/gsm8k_logs/{get_timestamp()}_{random_seed}"
    print(f"Running GSM8K 5 shots, log folder: {gsm8k_log_folder}")
    exec_command(f"mkdir -p {gsm8k_log_folder}")
    exec_command(_venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k "
        f"--num_fewshot {num_shots} "
        f"--apply_chat_template "
        f"--limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {random_seed} "
        f"--log_samples "
        f"--output_path {gsm8k_log_folder} "
        f"> {gsm8k_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def run_gsm8k_5_shots(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 100,
):
    if not os.path.exists(f"{LOG_DIR}/gsm8k_5_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/gsm8k_5_logs")
    random_seed = get_random_int()
    gsm8k_log_folder = f"{LOG_DIR}/gsm8k_5_logs/{get_timestamp()}_{random_seed}"
    print(f"Running GSM8K 5 shots, log folder: {gsm8k_log_folder}")
    exec_command(f"mkdir -p {gsm8k_log_folder}")
    exec_command(_venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k_cot_5shot "
        f"--num_fewshot 5 "
        f"--apply_chat_template "
        f"--limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {random_seed} "
        f"--log_samples "
        f"--output_path {gsm8k_log_folder} "
        f"> {gsm8k_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def run_gsm8k_100_shots(
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 50000,
    max_samples: Annotated[int, typer.Option()] = 100,
):
    if not os.path.exists(f"{LOG_DIR}/gsm8k_100_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/gsm8k_100_logs")
    random_seed = get_random_int()
    gsm8k_log_folder = f"{LOG_DIR}/gsm8k_100_logs/{get_timestamp()}_{random_seed}"
    print(f"Running GSM8K 100 shots, log folder: {gsm8k_log_folder}")
    exec_command(f"mkdir -p {gsm8k_log_folder}")
    exec_command(_venv_cmd(
        LMEVAL_VENV,
        f"nohup lm_eval "
        f"--model local-chat-completions "
        f'--model_args "model={MODEL_PATH},base_url=http://{HOST}:{PORT}/v1/chat/completions,num_concurrent=64,timeout=17280000" '
        f"--tasks gsm8k_cot_100shot "
        f"--num_fewshot 100 "
        f"--apply_chat_template "
        f"--limit {max_samples} "
        f'--gen_kwargs "max_tokens={max_tokens},temperature={temperature},top_p=0.95" '
        f"--seed {random_seed} "
        f"--log_samples "
        f"--output_path {gsm8k_log_folder} "
        f"> {gsm8k_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def setup_longbench():
    exec_command(f"uv venv {LONGBENCH_VENV}")
    exec_command(
        f"cd /sgl-workspace && git clone https://github.com/Fridge003/LongBench.git"
    )
    exec_command(
        _venv_cmd(LONGBENCH_VENV, "cd /sgl-workspace/LongBench && uv pip install -r requirements.txt")
    )


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
    if not os.path.exists(f"{LOG_DIR}/longbench_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/longbench_logs")

    random_seed = get_random_int()
    longbench_log_folder = f"{LOG_DIR}/longbench_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {longbench_log_folder}")
    print(f"Running LongBench, log folder: {longbench_log_folder}")

    exec_command(_venv_cmd(
        LONGBENCH_VENV,
        f"cd /sgl-workspace/LongBench && "
        f"HACK_THINKING={thinking} "
        f"HACK_SKIP_TOO_LONG_PROMPT={skip_too_long_prompt} "
        f"HACK_MAX_NEW_TOKENS={max_new_tokens} "
        f"HACK_MAX_LEN={max_len} "
        f"HACK_TEMPERATURE={temperature} "
        f"nohup python3 pred.py "
        f"--model {MODEL_PATH} "
        f"--n_proc {n_proc} "
        f"--num_prompts {num_prompts} "
        f"--port {PORT} "
        f"--save_dir {longbench_log_folder} "
        f"> {longbench_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def setup_ns():
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    if HF_TOKEN is None:
        raise ValueError("Please set HF_TOKEN for nemo skill setup")
    exec_command(f"uv venv {NS_VENV}")
    exec_command(_venv_cmd(
        NS_VENV,
        "uv pip install git+https://github.com/NVIDIA-NeMo/Skills.git@d77caab --ignore-installed blinker",
    ))
    exec_command(_venv_cmd(NS_VENV, f"HF_TOKEN={HF_TOKEN} ns prepare_data mmlu aime24 aime25"))
    # User might be asked for access of GPQA dataset. Just click in the hugging face website to grant access.
    exec_command(_venv_cmd(NS_VENV, f"HF_TOKEN={HF_TOKEN} ns prepare_data gpqa --split diamond"))


@app.command()
def run_gpqa(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    if not os.path.exists(f"{LOG_DIR}/gpqa_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/gpqa_logs")

    random_seed = get_random_int()
    gpqa_log_folder = f"{LOG_DIR}/gpqa_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {gpqa_log_folder}")
    print(f"Running GPQA, log folder: {gpqa_log_folder}")

    exec_command(_venv_cmd(
        NS_VENV,
        f"nohup ns eval "
        f"--server_type=openai "
        f"--model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=gpqa:{num_repeats} "
        f"--output_dir={gpqa_log_folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} "
        f"++inference.top_p=0.95 "
        f"++inference.timeout=25000000 "
        f"--starting_seed {random_seed} "
        f"> {gpqa_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def run_mmlu(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
    max_samples: Annotated[int, typer.Option()] = -1,
):
    if not os.path.exists(f"{LOG_DIR}/mmlu_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/mmlu_logs")

    random_seed = get_random_int()
    mmlu_log_folder = f"{LOG_DIR}/mmlu_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {mmlu_log_folder}")
    print(f"Running MMLU, log folder: {mmlu_log_folder}")

    bench_cmd = (
        f"--server_type=openai "
        f"--model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=mmlu:{num_repeats} "
        f"--output_dir={mmlu_log_folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} "
        f"++inference.top_p=0.95 "
        f"++inference.timeout=25000000 "
        f"--starting_seed {random_seed} "
    )

    if max_samples > 0:
        bench_cmd += f"++max_samples={max_samples} "

    exec_command(_venv_cmd(
        NS_VENV,
        f"nohup ns eval {bench_cmd} > {mmlu_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def run_aime25(
    num_repeats: Annotated[int, typer.Option()] = 16,
    temperature: Annotated[float, typer.Option()] = 1.0,
    max_tokens: Annotated[int, typer.Option()] = 60000,
    max_concurrency: Annotated[int, typer.Option()] = 64,
):
    if not os.path.exists(f"{LOG_DIR}/aime25_logs"):
        exec_command(f"mkdir -p {LOG_DIR}/aime25_logs")

    random_seed = get_random_int()
    aime25_log_folder = f"{LOG_DIR}/aime25_logs/{get_timestamp()}_{random_seed}"
    exec_command(f"mkdir -p {aime25_log_folder}")
    print(f"Running AIME25, log folder: {aime25_log_folder}")

    exec_command(_venv_cmd(
        NS_VENV,
        f"nohup ns eval "
        f"--server_type=openai "
        f"--model={MODEL_PATH} "
        f"--server_address=http://{HOST}:{PORT}/v1 "
        f"--benchmarks=aime25:{num_repeats} "
        f"--output_dir={aime25_log_folder} "
        f"++inference.tokens_to_generate={max_tokens} "
        f"++max_concurrent_requests={max_concurrency} "
        f"++inference.temperature={temperature} "
        f"++inference.top_p=0.95 "
        f"++inference.timeout=25000000 "
        f"--starting_seed {random_seed} "
        f"> {aime25_log_folder}/output.log 2>&1 &",
    ))


@app.command()
def exec_command(cmd: str, capture_output: bool = False) -> str | None:
    print(f"EXEC: {cmd}", flush=True)
    return subprocess.run(
        ["bash", "-c", cmd],
        shell=False,
        check=True,
        capture_output=capture_output,
        **(dict(text=True) if capture_output else {}),
    )


if __name__ == "__main__":
    app()