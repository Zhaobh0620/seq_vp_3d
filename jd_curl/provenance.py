"""Shared source snapshots and local runtime evidence; no external project imports."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile


def source_snapshot(output_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [root/"pyproject.toml", root/"README.md"]
    for folder, pattern in (("jd_curl", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), ("scripts", "*.sh"),
                            ("docs", "*.md"), ("configs", "*.json")):
        paths.extend(sorted((root/folder).rglob(pattern)))
    hashes = {}
    with tarfile.open(output_dir/"source.tar.gz", "x:gz") as archive:
        for path in paths:
            name = path.relative_to(root).as_posix()
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name)
    with (output_dir/"sha256.json").open("x") as stream:
        json.dump(hashes, stream, indent=2)
        stream.write("\n")
    return hashes


def file_provenance(path: Path) -> dict:
    path = Path(path).absolute()
    return dict(path=str(path), resolved_path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def runtime_metadata() -> dict:
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True) if git.returncode == 0 else None
    cpu = platform.processor()
    cpu_file = Path("/proc/cpuinfo")
    if cpu_file.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpu_file.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        executable=sys.executable, platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count(),
        versions={name: version(name) for name in ("numpy", "scipy")},
        thread_environment={key: os.environ.get(key) for key in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        git_head=git.stdout.strip() if git.returncode == 0 else None,
        git_status=status.stdout if status else None, git_note=git.stderr.strip(),
        code_identity="source.tar.gz + sha256.json",
        execution="solvers are serial within this run; system-wide concurrent activity is not controlled")
