"""Fingerprints, durable journal and the raw-only reconstruction interface."""
import runtime
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_hash(value):
    """Hash dtype, shape and bytes, not container/compression details."""
    a = np.ascontiguousarray(value)
    d = hashlib.sha256()
    d.update(str(a.dtype).encode())
    d.update(json.dumps(list(a.shape)).encode())
    d.update(a.tobytes())
    return d.hexdigest()


def tensor_digest(items):
    return canonical_hash({k: array_hash(v.detach().cpu().numpy())
                           for k, v in sorted(items)})


def model_fingerprints(model, checkpoint):
    return {"checkpoint_sha256": runtime.sha256(checkpoint),
            "CNN_parameters_sha256": tensor_digest(model.named_parameters()),
            "physics_buffers_sha256": tensor_digest(model.physics.named_buffers()),
            "all_buffers_sha256": tensor_digest(model.named_buffers()),
            "complete_config_sha256": canonical_hash(model.cfg),
            "config": model.cfg}


def case_identity(*, config_hash, case_id, fixture_hash, method, input_kind, input_hash, run_id):
    """Container equality does not imply task equality (oracle vs raw-DN paths)."""
    return canonical_hash({"config_sha256": config_hash, "case_id": case_id,
                           "fixture_sha256": fixture_hash, "method": method,
                           "input_kind": input_kind, "actual_input_sha256": input_hash,
                           "run_id": run_id})


@dataclasses.dataclass(frozen=True, slots=True)
class MeasurementInput:
    """Only deployable raw DN. No truth, split, seed or case metadata."""
    raw_dn: np.ndarray

    def __post_init__(self):
        if not isinstance(self.raw_dn, np.ndarray) or self.raw_dn.ndim != 2:
            raise ValueError("A single H x W raw-DN array is required")


def measurement_input(record):
    if set(record) != {"raw_dn"}:
        raise ValueError("Reconstructor interface accepts raw_dn only")
    return MeasurementInput(record["raw_dn"])


class Journal:
    """Append and fsync each outcome; only exact successful identities resume."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_rows, self._cached_size = [], -1

    def append(self, row):
        self.rows()  # synchronize any preceding external append before this write
        text = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        self._cached_rows.append(row)
        self._cached_size = self.path.stat().st_size

    def rows(self):
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size != self._cached_size:
            self._cached_rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._cached_size = size
        return list(self._cached_rows)

    def succeeded(self, identity):
        return any(r.get("identity_sha256") == identity and r.get("status") == "SUCCESS"
                   for r in self.rows())


def execute_record(journal, base, work):
    """A bad frame cannot discard earlier completed frames."""
    try:
        result = work()
        row = {**base, **result}
    except TimeoutError as exc:
        row = {**base, "status": "TIMEOUT", "error": str(exc), "metrics": None}
    except Exception as exc:
        row = {**base, "status": "EXCEPTION", "error": type(exc).__name__+": "+str(exc),
               "metrics": None}
    journal.append(row)
    return row


def outcome_counts(rows, registered):
    # Latest attempt per exact identity; earlier failure records remain in the journal.
    latest = {r["identity_sha256"]: r for r in rows}
    completed = sum(r["status"] == "SUCCESS" for r in latest.values())
    return {"registered": registered, "attempted": len(latest), "attempt_records": len(rows),
            "completed": completed, "failed_or_incomplete": len(latest)-completed,
            "remaining": max(0, registered-len(latest)),
            "status": "COMPLETE" if completed == registered else "PARTIAL"}
