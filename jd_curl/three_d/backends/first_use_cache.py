"""FC1 (throughput-v36): a disk cache of FX1 F2's first-use registry, so that the
dry pass before the task clock runs once per machine and grid instead of once per run. Registration results are unchanged:
the registry only lists keys that a dry pass on a synthetic field of this grid verified bit for bit; whether it came from
this process or from the cache, the run seeds the same keys.

The cache key is the sha256 of the key inputs, and a hit requires the stored inputs to equal the recomputed ones exactly:
- source: sha256 over (relative path, sha256 of the bytes) of every .py under jd_curl/ and the two run scripts of the tree
  the run executes (its frozen snapshot);
- device: GPU name and compute capability, total memory, torch / CUDA / triton / Python versions (the compiled binaries);
- geometry: the native grid's shape, spacing and origin from the run's manifest;
- profile: the throughput profile dict; options: the worker options of the run except the export reserve.
A cache file is written only from a dry pass that verified some family and raised nothing (atomic replace). Root:
JD_CURL_FIRST_USE_CACHE, else ~/.cache/seq_vp_3d/first_use."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

CACHE_POLICY = "first-use-registry-disk-cache-v1"
ENV = "JD_CURL_FIRST_USE_CACHE"


def cache_root():
    root = os.environ.get(ENV)
    return Path(root) if root else Path.home()/".cache"/"seq_vp_3d"/"first_use"


def source_digest(root):
    root = Path(root)
    files = sorted(p for p in (root/"jd_curl").rglob("*.py") if "__pycache__" not in p.parts)
    files += [root/"scripts"/"run_stage_3d_e19_mri.py", root/"scripts"/"run_stage_3d_e48_pg.py"]
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode()+b"\0"+hashlib.sha256(p.read_bytes()).hexdigest().encode()+b"\n")
    return h.hexdigest(), len(files)


def device_identity():
    import torch
    import triton
    props = torch.cuda.get_device_properties(0)
    return dict(name=props.name, capability=list(torch.cuda.get_device_capability(0)), total_memory=int(props.total_memory),
                torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__, python=platform.python_version())


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def key_inputs(root, manifest, profile, options):
    digest, n = source_digest(root)
    return _plain(dict(policy=CACHE_POLICY, source=digest, source_files=n, device=device_identity(),
                       geometry={k: manifest[k] for k in ("shape_zyx", "spacing_xyz", "origin_xyz")},
                       profile=profile, options={k: v for k, v in options.items() if k != "export_reserve"}))


def key_of(inputs):
    return hashlib.sha256(json.dumps(_plain(inputs), sort_keys=True).encode()).hexdigest()


def _registry_digest(registry):
    return hashlib.sha256(json.dumps(_plain(registry), sort_keys=True).encode()).hexdigest()


def load(inputs, root=None):
    """(registry, record) on a valid hit; (None, record with the reason) otherwise."""
    key = key_of(inputs)
    path = (Path(root) if root is not None else cache_root())/f"{key}.json"
    record = dict(policy=CACHE_POLICY, key=key, path=str(path), hit=False)
    if not path.exists():
        return None, dict(record, reason="absent")
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return None, dict(record, reason=f"unreadable: {type(exc).__name__}")
    if data.get("policy") != CACHE_POLICY or _plain(data.get("inputs")) != _plain(inputs):
        return None, dict(record, reason="inputs differ")
    registry = data.get("registry") or {}
    if not registry.get("families") or _registry_digest(registry) != data.get("registry_sha256"):
        return None, dict(record, reason="registry empty or digest mismatch")
    return registry, dict(record, hit=True, created_utc=data.get("created_utc"), dry_pass=data.get("dry_pass"))


def store(inputs, registry, dry_pass, root=None):
    key = key_of(inputs)
    folder = Path(root) if root is not None else cache_root()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"{key}.json"
    data = dict(policy=CACHE_POLICY, inputs=_plain(inputs), registry=_plain(registry), registry_sha256=_registry_digest(registry),
                dry_pass=_plain(dry_pass), created_utc=datetime.now(timezone.utc).isoformat())
    temp = folder/f"{key}.json.partial"
    temp.write_text(json.dumps(data, indent=1, sort_keys=True))
    temp.replace(path)
    return dict(stored=True, path=str(path), key=key)
