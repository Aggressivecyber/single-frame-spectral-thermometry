"""Keep generated files and caches in the checkout (or an explicit runtime root)."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("THERMOMETRY_RUNTIME", ROOT / "_runtime")).resolve()
sys.dont_write_bytecode = True
for name in ("tmp", "torch", "cache", "mpl"):
    (RUNTIME / name).mkdir(parents=True, exist_ok=True)
for key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[key] = str(RUNTIME / "tmp")
os.environ["TORCH_HOME"] = str(RUNTIME / "torch")
os.environ["XDG_CACHE_HOME"] = str(RUNTIME / "cache")
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "mpl")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "2")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_output(path):
    return Path(path).resolve()


def write_json(path, value):
    path = check_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
