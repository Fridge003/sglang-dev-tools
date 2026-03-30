# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
If Claude finds any error in the commands here, please update it instantly.

## What is SGLang?

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models. It features RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, and broad hardware support (NVIDIA, AMD, Intel, TPU, Ascend NPU).

## Modify a branch locally and push

### Fetch
First fetch the branch:
```bash
# If it's a branch in origin
git fetch origin
git checkout <branch-name>

# If it's from a pr
gh pr checkout <pr-number>
```

### Lint
To fix lint for the current branch:
```bash
# If pre-commit is not installed
pre-commit install

pre-commit run --all-files
```
Then commit the changed files.

### Push
When pushing to the remote, if we are pushing to a branch under origin remote,
```bash
git push origin HEAD:<branch-name>
```
will be good.

But if we are pushing to a PR branch, please first add the remote repo of that PR with
```bash
git remote add push-pr <repo-url>
```
Then push with
```bash
git push push-pr
```
Finally cleanup this tmp remote
```bash
git remote remove push-pr
```

## How to set up testing environemnt 

### Step 1: Log into machine
```bash
# H200
xxx

# B200
xxx
```

If there is any local change needed to be synced, please sync with:
```bash
sgldev ssh sync b200 # Can also be h200, depends on the machine specified
```

### Step 2: Create docker container `sglang_baizhou` if it isn't created
First install the development tool kits:
```bash
pip install --force-reinstall git+https://github.com/Fridge003/sglang-dev-tools --break-system-packages
```

If the docker has been created, just exec into that container:
```bash
sgldev docker exec
```
Else launch the docker `sglang_baizhou` with
```bash
sgldev docker create
```


### Step 4: Setup environment inside `sglang_baizhou` container
```bash
pip install --force-reinstall git+https://github.com/Fridge003/sglang-dev-tools

# Check the login (setup) status 
gh auth status

# If the user has logged in, no need to do anything
# If there is no user logged in, then run following commands.
sgldev dev setup-sglang <github-key>
```

### Step 5: Then execute the test commands as you wish. Remember to create a new worktree for modified codes.
Each time when you checkout to a different branch, remember to reinstall the sglang environment through
```bash
pip install -e python
```
The sgl_kernel package should be reinstalled through
```bash
pip install sglang_kernel --force-reinstall
```
If out-of-memory happens due to GPU occupied by other users, shift to empty GPUs with `CUDA_VISIBLE_DEVICES` environ.



## Build & Install

```bash
pip install -e "python"
pip install -e "python[test]" # For some testing dependencies
```

The pyproject.toml is at `python/pyproject.toml`. Hardware-specific variants exist: `pyproject_cpu.toml`, `pyproject_npu.toml`, `pyproject_xpu.toml`.

sgl-kernel (AOT CUDA/C++ kernels) is a separate package in `sgl-kernel/` with its own build system (CMake).

## Running the Server

```bash
sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct
killall_sglang    # Kill all running servers
```

## Testing (inside docker conatiner on GPU devices)

```bash
# Single test file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# JIT kernel test
python3 python/sglang/jit_kernel/tests/test_add_constant.py

# Run a CI suite locally
python3 test/run_suite.py --hw cpu --suite stage-a-test-cpu
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small
```

Tests use `unittest` or `pytest` but must end with `unittest.main()` or `pytest.main([__file__])`. The CI runner appends `-f` (failfast). Always use `CustomTestCase` (never raw `unittest.TestCase`). See `test/README.md` and the `/write-sglang-test` skill for full details.

### CI Registration

Every CI test file must register at module level with literal values (AST-parsed):

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small")
```

### CI Pipeline

Three sequential stages: **A** (pre-flight, ~3 min) -> **B** (basic, ~30 min) -> **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. See the `/ci-workflow-guide` skill for details.

### Test Model Constants

- Small (1B): `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (Llama-3.2-1B-Instruct)
- Standard (8B): `DEFAULT_MODEL_NAME_FOR_TEST` (Llama-3.1-8B-Instruct)
- MoE: `DEFAULT_MOE_MODEL_NAME_FOR_TEST` (Mixtral-8x7B-Instruct)

Defined in `python/sglang/test/test_utils.py`.

## Linting & Formatting

Pre-commit hooks handle formatting. Key tools:
- **black** (v26.1.0) - Python formatting
- **isort** (v7.0.0) - Import sorting
- **ruff** (v0.15.1) - Linting (F401 unused imports, F821 undefined names)
- **clang-format** (v20.1.7) - C++/CUDA formatting
- **codespell** - Spell checking

```bash
pre-commit run --all-files    # Run all hooks
pre-commit run ruff --all-files  # Run specific hook
```

## Architecture

### Process Model (LLM Runtime)

The LLM runtime (`python/sglang/srt/`) uses multiprocessing with ZMQ for IPC:

```
HTTP/gRPC Request
    -> Engine (python/sglang/srt/entrypoints/engine.py)
        -> TokenizerManager (main process) -- tokenizes requests
            -> Scheduler (subprocess, srt/managers/scheduler.py) -- batches & runs inference
                -> ModelRunner (srt/model_executor/model_runner.py) -- forward pass
                -> RadixCache (srt/mem_cache/radix_cache.py) -- prefix caching
            -> DetokenizerManager (subprocess) -- converts tokens back to text
        -> HTTP response
```

### Key Directories

| Path | Purpose |
|------|---------|
| `python/sglang/srt/` | **SGLang Runtime** - core inference engine |
| `python/sglang/srt/entrypoints/` | Server entry points (engine, HTTP, gRPC, OpenAI-compatible API) |
| `python/sglang/srt/managers/` | TokenizerManager, Scheduler, DetokenizerManager |
| `python/sglang/srt/model_executor/` | Model forward pass, CUDA graphs |
| `python/sglang/srt/models/` | 170+ model implementations (Llama, Qwen, DeepSeek, etc.) |
| `python/sglang/srt/layers/` | Attention, linear, MoE, quantization, sampling layers |
| `python/sglang/srt/mem_cache/` | RadixCache, memory pools, hierarchical caching |
| `python/sglang/srt/configs/` | Model configs, server args (`server_args.py` is the main config) |
| `python/sglang/srt/distributed/` | Tensor/Pipeline/Expert parallelism |
| `python/sglang/srt/disaggregation/` | Prefill-decode disaggregation |
| `python/sglang/srt/speculative/` | EAGLE, n-gram speculative decoding |
| `python/sglang/srt/lora/` | Multi-LoRA adapter management |
| `python/sglang/srt/constrained/` | Structured output (xgrammar, llguidance, outlines) |
| `python/sglang/lang/` | Frontend DSL (gen, select, function) and backend adapters |
| `python/sglang/jit_kernel/` | Lightweight JIT Triton kernels |
| `python/sglang/multimodal_gen/` | Diffusion/multimodal generation (separate subsystem, has its own CLAUDE.md) |
| `sgl-kernel/` | Heavyweight AOT CUDA/C++ kernels (separate CMake build) |
| `sgl-model-gateway/` | Rust-based model gateway service |
| `test/registered/` | CI test files (auto-discovered by run_suite.py) |
| `test/manual/` | Non-CI tests for local debugging |
| `benchmark/` | Performance benchmarks |

### Scheduling & Batching

The Scheduler (`srt/managers/scheduler.py`) is the core orchestrator:
- Scheduling policies in `schedule_policy.py`: LPM, DFS, FCFS, LOF, RANDOM
- Batch structures in `schedule_batch.py`: request lifecycle and batch composition
- Mixins for pipeline parallelism, DP attention, weight updates, output processing

### IPC & Data Structures

`python/sglang/srt/managers/io_struct.py` defines all inter-process RPC request/response types. ZMQ sockets handle communication between tokenizer, scheduler, and detokenizer processes.

## Available Claude Code Skills

- `/write-sglang-test` - Templates, fixtures, model selection, CI registration for tests
- `/ci-workflow-guide` - CI pipeline internals, debugging CI failures
- `/add-jit-kernel` - Adding lightweight JIT Triton kernels
- `/add-sgl-kernel` - Adding heavyweight AOT CUDA/C++ kernels to sgl-kernel
- `/debug-cuda-crash` - Debugging CUDA crashes with kernel API logging
- `/generate-profile` - E2E profiling trace generation
- `/sglang-bisect-ci-regression` - Bisecting CI regressions

## sglang-dev-tools (`sgldev`) CLI Reference

Installed via `pip install --force-reinstall git+https://github.com/Fridge003/sglang-dev-tools`. Entry point: `sgldev`.

### `sgldev dev` -- Development Helpers

| Command | Description |
|---|---|
| `sgldev dev setup-sglang <GITHUB_TOKEN>` | Clone sglang, install pre-commit hooks, pip install, configure git, authenticate gh CLI. Options: `--name`, `--email` |
| `sgldev dev setup-cursor` | Install Cursor IDE extensions (Python, GitLens) |
| `sgldev dev download-model <MODEL_PATH>` | Download a model from HuggingFace Hub. Options: `--hf-token` |
| `sgldev dev kill-action <ACTION_ID>` | Force-cancel a GitHub Actions workflow run on sgl-project/sglang |
| `sgldev dev create-uv` | Create a uv virtual environment at `/sgl-workspace/sgl`. Options: `--python` (default 3.12) |

### `sgldev docker` -- Docker Container Management

| Command | Description |
|---|---|
| `sgldev docker create` | Create and start a new SGLang dev container. Options: `--name`, `--image`, `--cache-path`, `--hf-token`, `--shm-size` (32g), `--gpus`, `-v`, `-e`, `--ptrace` |
| `sgldev docker exec` | Exec into a running container. Options: `--name`, `--shell` |
| `sgldev docker pull` | Pull the latest image from registry |
| `sgldev docker rm` | Remove a container. Options: `--name`, `--force` |
| `sgldev docker logs <NAME>` | Show container logs. Options: `-f`, `--tail` |
| `sgldev docker killer` | Launch a privileged container with host PID namespace (for debugging) |
| `sgldev docker list` | List containers. Options: `--all/-a`, `--filter-name` |

### `sgldev server` -- Server Management

| Command | Description |
|---|---|
| `sgldev server health` | Check if the SGLang server is healthy. Options: `--host`, `--port` |
| `sgldev server flush` | Flush all KV cache contents. Options: `--host`, `--port` |
| `sgldev server kill` | Kill SGLang server process(es). Options: `--port` (default 30000) |

#### `sgldev server launch` -- Model-Specific Server Launch

All launch commands support: `--dp` (data parallelism), `--mtp` (speculative decoding), `--host`, `--port`, `--tee-log`.

| Command | Model | Default TP |
|---|---|---|
| `sgldev server launch dsv32` | `deepseek-ai/DeepSeek-V3.2` | TP=8 |
| `sgldev server launch dsv32-fp4` | `nvidia/DeepSeek-V3.2-NVFP4` (quantization: modelopt_fp4) | TP=4 |
| `sgldev server launch glm5` | `zai-org/GLM-5-FP8` | TP=8 |
| `sgldev server launch kimi25` | `moonshotai/Kimi-K2.5` | TP=8 |
| `sgldev server launch qwen3-8b` | `Qwen/Qwen3-8B` | TP=1 |
| `sgldev server launch qwen3-30b` | `Qwen/Qwen3-30B-A3B` | TP=1 |

### `sgldev acc` -- Accuracy Evaluation Benchmarks

#### Setup

| Command | Description |
|---|---|
| `sgldev acc setup-lmeval` | Set up lm-evaluation-harness in a dedicated venv |
| `sgldev acc setup-ns` | Set up NVIDIA NeMo Skills (requires `HF_TOKEN`; prepares MMLU, AIME24, AIME25, GPQA data) |
| `sgldev acc setup-longbench` | Set up LongBench in a dedicated venv |

#### Evaluations

| Command | Key Options | Description |
|---|---|---|
| `sgldev acc send-one` | `--port` | Send a single test request to verify server is up |
| `sgldev acc sgl-gsm8k` | `--num-shots` (8) | Run GSM8K via bench_sglang.py (1319 questions) |
| `sgldev acc run-gsm8k` | `--temperature`, `--max-tokens`, `--max-samples`, `--num-shots` (5) | Run GSM8K via lm-eval |
| `sgldev acc run-gsm8k-5-shots` | `--max-samples` (100) | GSM8K CoT 5-shot evaluation |
| `sgldev acc run-gsm8k-100-shots` | `--max-samples` (100) | GSM8K CoT 100-shot evaluation |
| `sgldev acc run-gpqa` | `--num-repeats` (16), `--temperature` (1.0) | GPQA evaluation via NeMo Skills |
| `sgldev acc run-mmlu` | `--num-repeats` (16), `--max-samples` | MMLU evaluation via NeMo Skills |
| `sgldev acc run-aime25` | `--num-repeats` (16) | AIME25 evaluation via NeMo Skills |
| `sgldev acc run-longbench` | `--n-proc`, `--num-prompts`, `--thinking` | LongBench evaluation |

### `sgldev bench` -- Benchmarking

| Command | Description |
|---|---|
| `sgldev bench serve` | Run `sglang.bench_serving` with random data. Options: `--num-prompts`, `--input-len`, `--output-len`, `--max-concurrency` |

### `sgldev profile` -- Profiling

| Command | Description |
|---|---|
| `sgldev profile one-batch` | Single-batch latency profiling via `sglang.bench_one_batch_server`. Options: `--model`, `--batch-size`, `--input-len`, `--output-len`, `--profile-steps`, `--profiler-dir` |
| `sgldev profile serving` | Throughput profiling via `sglang.bench_serving`. Options: `--backend`, `--num-prompts`, `--dataset-name`, `--max-concurrency` |

### `sgldev ssh` -- SSH and Rsync

Servers stored in `~/.config/sgldev/servers.json`.

| Command | Description |
|---|---|
| `sgldev ssh server-set <NAME>` | Add/update a server alias. Options: `--host` (required), `--user`, `--port`, `--key` |
| `sgldev ssh server-rm <NAME>` | Remove a named server |
| `sgldev ssh server-ls` | List all saved servers |
| `sgldev ssh connect <SERVER>` | SSH into a server. Options: `--cmd` (one-off command). Extra flags after `--` forwarded to ssh |
| `sgldev ssh rsync <SRC> <DST>` | Rsync files. Options: `--server`, `--to-remote`, `--delete`, `--dry-run`, `--exclude` |