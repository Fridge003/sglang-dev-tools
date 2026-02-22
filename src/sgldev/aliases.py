"""Server alias management backed by a local JSON file (~/.config/sgldev/servers.json).

The file stores a list of dicts, each with an "alias" key plus connection fields:

    [
      {"alias": "devbox", "host": "10.0.0.1", "port": 22, "key": "~/.ssh/id_rsa"},
      {"alias": "prod",   "host": "10.0.0.2", "port": 2222}
    ]
"""

import json
from pathlib import Path

ALIASES_FILE = Path.home() / ".config" / "sgldev" / "servers.json"


def _load() -> list[dict]:
    if not ALIASES_FILE.exists():
        return []
    return json.loads(ALIASES_FILE.read_text())


def _save(data: list[dict]) -> None:
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _find(data: list[dict], alias: str) -> int:
    """Return the index of the entry with the given alias, or -1."""
    for i, entry in enumerate(data):
        if entry.get("alias") == alias:
            return i
    return -1


def resolve(alias: str) -> dict[str, str | int]:
    """Return the server entry for *alias*, or exit with an error."""
    data = _load()
    idx = _find(data, alias)
    if idx == -1:
        available = ", ".join(e["alias"] for e in data) or "(none)"
        raise SystemExit(
            f"Unknown alias '{alias}'. Available: {available}\n"
            f"Run 'sgldev ssh alias-set <name> --host <ip>' to create one."
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
    entry: dict[str, str | int] = {"alias": name, "host": host}
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
        available = ", ".join(e["alias"] for e in data) or "(none)"
        raise SystemExit(f"Alias '{name}' not found. Available: {available}")
    data.pop(idx)
    _save(data)


def list_all() -> list[dict]:
    return _load()
