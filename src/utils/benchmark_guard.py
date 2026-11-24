"""Locked-benchmark protection. Phases 0-5 may only see path/checksum/size of the locked manifest."""
import hashlib
import os
from pathlib import Path

LOCKED_NAMES = ("evaluation_manifest",)
PHASE6_ENV = "DELTANET_PHASE6_UNLOCK"  # set to "1" only inside benchmark_runner (Phase 6)


def is_locked(path: str) -> bool:
    return any(n in Path(str(path)).name for n in LOCKED_NAMES)


def assert_not_locked(path: str) -> None:
    if is_locked(path) and os.environ.get(PHASE6_ENV) != "1":
        raise PermissionError(f"Locked benchmark '{path}' may only be loaded in Phase 6.")


def manifest_metadata(path: str) -> dict:
    """Existence, size and SHA-256 only. Never parses contents."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(p), "exists": True, "bytes": p.stat().st_size, "sha256": h}
