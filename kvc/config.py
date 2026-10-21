import json
import os
from pathlib import Path


def load_config(path, overrides=()) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return cfg


def run_id(config_path) -> str:
    return os.environ.get("RUN_ID") or Path(config_path).stem
