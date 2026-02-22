# sglang-dev-tools (`sgldev`)

CLI toolkit for SGLang development, evaluation, profiling, and deployment.

## Install

```bash
pip install -e .
```

## Usage

```bash
sgldev --help              # Show all command groups
sgldev acc --help          # Accuracy evaluation benchmarks
sgldev server --help       # Server launch / health / kill
sgldev profile --help      # Profiling (one-batch latency, serving throughput)
sgldev docker --help       # Docker container management
sgldev ssh --help          # SSH connection and rsync operations
```

### Examples

```bash
# Launch an SGLang server with DeepSeek FP4
sgldev server launch \
  --model-path nvidia/DeepSeek-V3-0324-FP4 \
  --tp 4 --dp 4 --enable-dp-attention \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend trtllm_mla \
  --quantization modelopt_fp4

# Profile single-batch latency
sgldev profile one-batch --batch-size 16 --input-len 1024 --output-len 20

# Profile serving throughput
sgldev profile serving --num-prompts 64 --random-input 32000 --random-output 1024

# Run GSM8K accuracy eval
sgldev acc run-gsm8k --temperature 0.0 --max-tokens 50000 --num-shots 5

# Create a dev container
sgldev docker create --name sglang_dev --cache-path /data/hf-cache

# SSH into a remote machine
sgldev ssh connect --host 10.0.0.1

# Run a remote command over SSH
sgldev ssh connect --host 10.0.0.1 --cmd "nvidia-smi"

# Forward a port via SSH
sgldev ssh connect --host 10.0.0.1 -- -L 8080:localhost:8080

# Push local directory to remote with rsync
sgldev ssh rsync ./data /data --host 10.0.0.1

# Pull from remote with rsync
sgldev ssh rsync /data/results ./results --host 10.0.0.1 --no-to-remote
```

## Configuration

All modules read sensible defaults from environment variables:

| Variable | Default | Used by |
|---|---|---|
| `MODEL_PATH` | `/data/weights/hello2026` | acc |
| `CODE_PATH` | `/sgl-workspace/NightFall` | acc |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | acc, server |
| `HOST` | `127.0.0.1` / `0.0.0.0` | acc, server, profile |
| `PORT` | `30010` / `30000` | acc, server, profile |
| `LOG_DIR` | `/data/logs` | acc, server |
| `HF_TOKEN` | - | acc (setup-ns), docker |
| `SGDEV_DOCKER_IMAGE` | `lmsysorg/sglang:dev` | docker |
| `SGDEV_DOCKER_CACHE` | - | docker |
| `SSH_USER` | `radixark` | ssh |
| `SSH_HOST` | - | ssh |
| `SSH_KEY` | `~/.ssh/sglang_dev` | ssh |
| `SSH_PORT` | `22` | ssh |

## Adding a new command group

1. Create `src/sgldev/mymodule.py`:
   ```python
   import typer
   app = typer.Typer(no_args_is_help=True)

   @app.command()
   def my_command():
       ...
   ```
2. Register it in `src/sgldev/cli.py`:
   ```python
   from sgldev.mymodule import app as mymodule_app
   app.add_typer(mymodule_app, name="mymod", help="My new commands")
   ```
3. That's it. `sgldev mymod my-command` is now available.
