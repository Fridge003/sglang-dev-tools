"""Server management backed by a local JSON file (~/.config/sgldev/servers.json).

The file stores a list of dicts, each with a "server" key plus connection fields:

    [
      {"server": "devbox", "host": "10.0.0.1", "port": 22, "key": "~/.ssh/id_rsa"},
      {"server": "prod",   "host": "10.0.0.2", "port": 2222}
    ]
"""

import json
from pathlib import Path

SERVERS_FILE = Path.home() / ".config" / "sgldev" / "servers.json"


def _load() -> list[dict]:
    if not SERVERS_FILE.exists():
        return []
    return json.loads(SERVERS_FILE.read_text())


def _save(data: list[dict]) -> None:
    SERVERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SERVERS_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _find(data: list[dict], server: str) -> int:
    """Return the index of the entry with the given server name, or -1."""
    for i, entry in enumerate(data):
        if entry.get("server") == server:
            return i
    return -1


def resolve(server: str) -> dict[str, str | int]:
    """Return the server entry for *server*, or exit with an error."""
    data = _load()
    idx = _find(data, server)
    if idx == -1:
        available = ", ".join(e["server"] for e in data) or "(none)"
        raise SystemExit(
            f"Unknown server '{server}'. Available: {available}\n"
            f"Run 'sgldev ssh server-set <name> --host <ip>' to create one."
        )
    return data[idx]


def add(
    name: str,
    host: str,
    user: str | None = None,
    port: int | None = None,
    key: str | None = None,
) -> None:
    data = _load()
    entry: dict[str, str | int] = {"server": name, "host": host}
    if user is not None:
        entry["user"] = user
    if port is not None:
        entry["port"] = port
    if key is not None:
        entry["key"] = key

    idx = _find(data, name)
    if idx != -1:
        data[idx] = entry
    else:
        data.append(entry)
    _save(data)


def remove(name: str) -> None:
    data = _load()
    idx = _find(data, name)
    if idx == -1:
        available = ", ".join(e["server"] for e in data) or "(none)"
        raise SystemExit(f"Server '{name}' not found. Available: {available}")
    data.pop(idx)
    _save(data)


def list_all() -> list[dict]:
    return _load()
