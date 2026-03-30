"""Experiment directories: manifest + JSONL iteration log."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    manifest: Path
    iterations: Path


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def create_experiment(task: str, dataset_path: str | Path, experiments_dir: str = "experiments") -> ExperimentPaths:
    """Create a new experiment directory with manifest.json."""
    dataset_path = Path(dataset_path).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:8]
    root = Path(experiments_dir) / f"{stamp}-{uid}"
    root.mkdir(parents=True, exist_ok=True)

    manifest = root / "manifest.json"
    iterations = root / "iterations.jsonl"

    ds_hash = _file_sha256(dataset_path) if dataset_path.is_file() else ""

    manifest.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "task": task,
                "dataset_path": str(dataset_path),
                "dataset_sha256": ds_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    iterations.write_text("", encoding="utf-8")

    return ExperimentPaths(root=root, manifest=manifest, iterations=iterations)


def append_iteration(paths: ExperimentPaths, record: dict[str, Any]) -> None:
    with paths.iterations.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
