from __future__ import annotations

import json
import os
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_run_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = payload.copy()
    enriched["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = payload.copy()
    canonical = json.dumps(enriched, sort_keys=True, ensure_ascii=False)
    enriched["manifest_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    enriched["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return enriched


def load_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
