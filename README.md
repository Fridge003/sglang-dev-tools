# sglang-dev-tools (`sgldev`)

CLI toolkit for SGLang development, evaluation, profiling, and deployment.

## Install

```bash
pip install -e .
pip install --force-reinstall git+https://github.com/Fridge003/sglang-dev-tools
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
