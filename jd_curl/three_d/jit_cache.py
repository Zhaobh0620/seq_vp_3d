"""E107 (throughput-v25 option jit_cache): a content-addressed on-disk cache for the Numba kernels of jd_curl.

Every worker process compiles the same Numba kernels again (about 30 s of the start-up for the recommended profile,
before the task clock). Numba's own cache keys entries by the source file's path and timestamp, and runs execute from
per-run source snapshots, so it would never be reused across runs (and it does not see changes of called functions in
other files). Here the cache directory is keyed instead by a SHA-256 of
  - every .py file of the running jd_curl tree (relative path and content),
  - the code objects of the kernel modules AS LOADED IN THIS PROCESS (every module-level function, every dispatcher's
    function and target options, every intrinsic's definition; co_code, co_consts recursively, co_names), so a tree
    edited between import and enable() can never file old code under the new tree's key (review E107 S1),
  - the Python, NumPy, Numba and llvmlite versions and the Numba configuration that shapes code generation,
  - the values of the constant globals the kernels name;
and inside that directory Numba's own index key adds the signature, the target (triple, CPU name, CPU features) and the
function's bytecode and closure. Every load is guarded (review E107 S2/S4/S5): a load error, a save error, or a loaded
entry whose signature is not the requested one is a miss (recorded), and the kernel compiles as without the cache.
Kernels on a call cycle (self- or mutual recursion) and their callers are never cached (loading cached recursive code
crashed Numba 0.65 in the E107 prototype); neither are dispatchers sharing a Python function, nor kernels built later by
factories. A cached entry is therefore the machine code an identical compile produced, so every value is the same.

enable(root) must run before any kernel compiles: it imports every kernel module (KERNEL_MODULES) and turns caching on
for each module-level dispatcher not compiled yet. Off unless called (throughput profiles with jit_cache only).
"""
import hashlib
import os
import sys
import tempfile
import types
from pathlib import Path

POLICY = "numba-content-addressed-cache-v2"
KERNEL_MODULES = ("jd_curl.three_d.bernstein_barrier", "jd_curl.three_d.certificate_prescreen",
                  "jd_curl.three_d.compiled_enclosure", "jd_curl.three_d.compiled_fallback", "jd_curl.three_d.fast_checks",
                  "jd_curl.three_d.host_kernels", "jd_curl.three_d.soa_enclosure",
                  "jd_curl.three_d.solvers.jd_affine_compiled")      # every jd_curl module defining Numba kernels
CONFIG_KEYS = ("OPT", "BOUNDSCHECK", "LOOP_VECTORIZE", "SLP_VECTORIZE", "ENABLE_AVX", "DISABLE_INTEL_SVML",
               "USING_SVML", "DEBUGINFO_DEFAULT", "CPU_NAME", "CPU_FEATURES", "DISABLE_JIT")
_STATE = dict(enabled=False, root=None, key=None, tree=None, dispatchers=[], skipped=[], refused=None,
              load_errors=0, save_errors=0, signature_mismatches=0, error_types=[], root_conflicts=0, index_repairs=0)


def default_root():
    """JD_CURL_JIT_CACHE, else ~/.cache/seq_vp_3d/numba_jit (or a temporary directory without a home)."""
    root = os.environ.get("JD_CURL_JIT_CACHE")
    if root:
        return root
    try:
        return str(Path("~/.cache/seq_vp_3d/numba_jit").expanduser())
    except RuntimeError:
        return os.path.join(tempfile.gettempdir(), "seq_vp_3d_numba_jit")


def _tree_root():
    return Path(__file__).resolve().parents[1]           # .../jd_curl


def tree_hash(root=None):
    """SHA-256 over (relative path, content) of every .py file under the jd_curl package, in sorted order; a file
    that cannot be read contributes its path and the marker 'unreadable' (review E107 M5)."""
    root = Path(root) if root is not None else _tree_root()
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def _versions():
    import numpy
    import numba
    import llvmlite
    from numba.core import config
    settings = ",".join(f"{k}={getattr(config, k, None)!r}" for k in CONFIG_KEYS)
    return "py%d.%d.%d-numpy%s-numba%s-llvmlite%s-" % (sys.version_info[:3]+(numpy.__version__, numba.__version__,
                                                                           llvmlite.__version__))+settings


def _code_digest(h, code, depth=0):
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if isinstance(const, types.CodeType) and depth < 8:
            _code_digest(h, const, depth+1)
        elif isinstance(const, frozenset):          # review E107 (round 2): repr order follows the hash seed
            h.update(repr(sorted(map(repr, const))).encode())
        else:
            h.update(repr(const).encode())


def _modules_code_digest(entries=None):
    """SHA-256 of the code of the kernel modules as loaded: module-level functions, dispatchers (function and target
    options) and intrinsics (definition function: _defn in Numba 0.65, or __wrapped__), in sorted order. entries: a
    list that receives the (module.attr, kind) pairs digested."""
    from numba.core.dispatcher import Dispatcher
    h = hashlib.sha256()
    for name in KERNEL_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, value in sorted(vars(module).items()):
            code = None
            if isinstance(value, Dispatcher) and getattr(value.py_func, "__module__", None) == name:
                code = value.py_func.__code__
                h.update(f"{name}.{attr}:dispatcher:{sorted(value.targetoptions.items())!r}".encode())
            elif isinstance(value, types.FunctionType) and value.__module__ == name:
                code = value.__code__
                h.update(f"{name}.{attr}:function".encode())
            else:
                definition = getattr(value, "_defn", None) or getattr(value, "__wrapped__", None)
                if isinstance(definition, types.FunctionType) and definition.__module__ == name:
                    code = definition.__code__
                    h.update(f"{name}.{attr}:intrinsic".encode())
            if code is not None:
                _code_digest(h, code)
                if entries is not None:
                    entries.append(f"{name}.{attr}")
    return h.hexdigest()


def _dispatchers():
    """(module name, attribute name, dispatcher) of every Numba dispatcher defined at module level in the kernel
    modules (review E107 round 2: exactly the modules whose loaded code the key digests)."""
    from numba.core.dispatcher import Dispatcher
    out = []
    for name in sorted(KERNEL_MODULES):
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, value in sorted(vars(module).items()):
            if isinstance(value, Dispatcher) and getattr(value.py_func, "__module__", None) == name:
                out.append((name, attr, value))
    return out


def _feed_constant(h, value):
    import numpy as np
    if isinstance(value, np.ndarray):
        h.update(f"array:{value.dtype.str}{value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (tuple, list)):
        h.update(b"(")
        for item in value:
            _feed_constant(h, item)
        h.update(b")")
    elif isinstance(value, (bool, int, float, complex, str, bytes, np.generic)) or value is None:
        h.update(f"{type(value).__name__}:{value!r}".encode())
    else:
        h.update(f"<{type(value).__name__}>".encode())


def _constant_globals(dispatchers):
    """SHA-256 of the non-callable global constants the kernels' bytecode names (numbers, NumPy scalars, strings,
    tuples and lists recursively, arrays by bytes)."""
    import numpy as np
    h = hashlib.sha256()
    for module, attr, d in dispatchers:
        g = d.py_func.__globals__
        for name in sorted(set(d.py_func.__code__.co_names)):
            if name not in g:
                continue
            v = g[name]
            if isinstance(v, (bool, int, float, complex, str, bytes, tuple, list, np.ndarray, np.generic)) or v is None:
                h.update(f"{module}.{attr}:{name}=".encode())
                _feed_constant(h, v)
                h.update(b"\n")
    return h.hexdigest()


def cache_key(dispatchers, tree=None):
    return hashlib.sha256("|".join((tree_hash(tree), _versions(), _modules_code_digest(),
                                    _constant_globals(dispatchers))).encode()).hexdigest()[:32]


def _recursive_closure(dispatchers):
    """(module, attr) of every kernel on a call cycle (self-recursion or mutual recursion through other kernels named
    in its module's globals), every kernel naming a dispatcher outside the kernel modules' set or an intrinsic defined
    outside the kernel modules (code the key does not cover), and every kernel that calls one of these, transitively;
    these are never cached."""
    from numba.core.dispatcher import Dispatcher
    by_id = {id(d): (module, attr) for module, attr, d in dispatchers}
    calls, outside = {}, set()
    for module, attr, d in dispatchers:
        g = d.py_func.__globals__
        calls[(module, attr)] = {by_id[id(g[n])] for n in d.py_func.__code__.co_names
                                 if n in g and isinstance(g[n], Dispatcher) and id(g[n]) in by_id}
        for n in d.py_func.__code__.co_names:     # review E107 (round 3): callees whose code the key does not cover
            v = g.get(n)
            definition = getattr(v, "_defn", None)
            if ((isinstance(v, Dispatcher) and id(v) not in by_id)
                    or (definition is not None and getattr(definition, "__module__", None) not in KERNEL_MODULES)):
                outside.add((module, attr))

    def reaches(start, target):
        seen, stack = set(), list(calls.get(start, ()))
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                stack.extend(calls.get(node, ()))
        return False
    bad = {key for key in calls if reaches(key, key)} | outside
    changed = True
    while changed:
        changed = False
        for key, callees in calls.items():
            if key not in bad and callees & bad:
                bad.add(key)
                changed = True
    return bad


def _note_error(kind, error):
    _STATE[kind] += 1
    name = type(error).__name__
    if name not in _STATE["error_types"]:
        _STATE["error_types"].append(name)


class _GuardedCache:
    """Numba's FunctionCache with every load and save guarded: any exception is a miss (load) or ignored (save), and a
    loaded entry whose signature is not the requested one is a miss; all are counted in the record."""

    def __init__(self, inner):
        self._inner = inner

    def load_overload(self, sig, target_context):
        from numba.core import sigutils
        try:
            cres = self._inner.load_overload(sig, target_context)
        except Exception as error:      # noqa: BLE001 - compiled as without the cache
            _note_error("load_errors", error)
            return None
        if cres is None:
            return None
        try:
            args, _ = sigutils.normalize_signature(sig)
            same = tuple(cres.signature.args) == tuple(args)
        except Exception as error:      # noqa: BLE001
            _note_error("load_errors", error)
            return None
        if not same:
            _STATE["signature_mismatches"] += 1
            return None
        return cres

    def save_overload(self, sig, cres):
        try:
            self._inner.save_overload(sig, cres)
        except Exception as error:      # noqa: BLE001 - the compiled code is used
            _note_error("save_errors", error)
            try:                        # a corrupt index makes every save fail: clear it (only then) and retry once
                self._inner._cache_file._load_index()
                return
            except Exception:           # noqa: BLE001 - the index itself is unreadable
                pass
            try:
                self._inner.flush()
                self._inner.save_overload(sig, cres)
                _STATE["index_repairs"] += 1
            except Exception as again:  # noqa: BLE001 - nothing saved
                _note_error("save_errors", again)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TreeLocator:
    """A Numba cache locator for jd_curl sources: one directory per cache key, the file's path inside the package and the
    function's first line as the disambiguator, and the key itself as the source stamp."""

    def __init__(self, py_func, py_file, relative):
        self._py_file = py_file                 # read by CompileResultCacheImpl.check_cachable (review E107 M1)
        self._relative = relative
        self._line = py_func.__code__.co_firstlineno
        self._path = os.path.join(_STATE["root"], _STATE["key"], os.path.dirname(relative) or ".")

    def ensure_cache_path(self):
        os.makedirs(self._path, exist_ok=True)
        tempfile.TemporaryFile(dir=self._path).close()

    def get_cache_path(self):
        return self._path

    def get_source_stamp(self):
        return _STATE["key"]

    def get_disambiguator(self):
        return hashlib.sha1(f"{self._relative}:{self._line}".encode()).hexdigest()[:12]

    @classmethod
    def from_function(cls, py_func, py_file):
        if not _STATE["enabled"]:
            return None
        try:
            relative = os.path.relpath(os.path.realpath(py_file), _STATE["tree"])
        except ValueError:
            return None
        if relative.startswith(".."):
            return None
        return cls(py_func, py_file, relative)


def _writable(root):
    try:
        os.makedirs(root, exist_ok=True)
        tempfile.TemporaryFile(dir=root).close()
        return True
    except OSError:
        return False


def enable(root):
    """Turn the content-addressed cache on for this process (idempotent; a second root is counted and ignored).
    Returns the record. Refuses (record 'refused') when NUMBA_CACHE_LOCATOR_CLASSES is set or the root is not
    writable; nothing is cached then."""
    root = os.path.abspath(os.path.expanduser(str(root)))
    if _STATE["enabled"] or _STATE["refused"]:
        if _STATE["root"] is not None and root != _STATE["root"]:
            _STATE["root_conflicts"] += 1
        return record()
    import importlib
    from numba.core import caching, config
    from numba.core.caching import _CacheLocator, NullCache
    if getattr(config, "CACHE_LOCATOR_CLASSES", None):
        _STATE["refused"] = "NUMBA_CACHE_LOCATOR_CLASSES is set"
        return record()
    if not _writable(root):
        _STATE["refused"] = "root not writable"
        return record()
    for name in KERNEL_MODULES:               # before any of their kernels compiles
        importlib.import_module(name)
    locator = type("JdCurlTreeLocator", (_TreeLocator, _CacheLocator), {})
    tree = _tree_root()
    dispatchers = _dispatchers()
    _STATE.update(enabled=True, root=root, key=cache_key(dispatchers, tree), tree=str(tree.resolve()))
    impl = getattr(caching, "CacheImpl", None) or getattr(caching, "_CacheImpl")      # numba 0.65: CacheImpl
    impl._locator_classes = [locator]+[c for c in impl._locator_classes if c.__name__ != "JdCurlTreeLocator"]
    excluded = _recursive_closure(dispatchers)
    counts = {}
    for _, _, d in dispatchers:
        counts[id(d.py_func)] = counts.get(id(d.py_func), 0)+1
    for module, attr, d in dispatchers:
        name = f"{module}.{attr}"
        if not isinstance(d._cache, NullCache):
            continue
        if (module, attr) in excluded:
            _STATE["skipped"].append(f"{name}: recursive")
            continue
        if counts[id(d.py_func)] > 1:
            _STATE["skipped"].append(f"{name}: shared python function")
            continue
        if d.overloads:                       # already compiled in this process: leave it as it is
            _STATE["skipped"].append(f"{name}: compiled before enable")
            continue
        try:
            d.enable_caching()
            if not isinstance(d._cache._impl.locator, locator):     # review E107 S3: never a stock locator
                d._cache = NullCache()
                _STATE["skipped"].append(f"{name}: not under the tree locator")
                continue
            d._cache = _GuardedCache(d._cache)
            _STATE["dispatchers"].append(name)
        except Exception as error:            # noqa: BLE001 - such a kernel compiles as before
            d._cache = NullCache()
            _STATE["skipped"].append(f"{name}: {type(error).__name__}")
    return record()


def enable_if(flag):
    """enable(default_root()) when flag (the profile's jit_cache), else None and nothing is read or written."""
    return enable(default_root()) if flag else None


def dispatcher_stats():
    """{module.attr: (hits, misses, compiled signatures)} for the cached dispatchers."""
    out = {}
    if not _STATE["enabled"]:
        return out
    names = set(_STATE["dispatchers"])
    for module, attr, d in _dispatchers():
        name = f"{module}.{attr}"
        if name in names:
            out[name] = (sum(d._cache_hits.values()), sum(d._cache_misses.values()), len(d.overloads))
    return out


def record():
    try:
        stats = dispatcher_stats()
        hits = sum(v[0] for v in stats.values())
        misses = sum(v[1] for v in stats.values())
    except Exception:                          # noqa: BLE001 - a record must never fail a run (review E107 M5)
        hits = misses = None
    return dict(policy=POLICY, enabled=_STATE["enabled"], refused=_STATE["refused"], root=_STATE["root"],
                key=_STATE["key"], dispatchers=len(_STATE["dispatchers"]), skipped=list(_STATE["skipped"]),
                cache_hits=hits, cache_misses=misses, load_errors=_STATE["load_errors"],
                save_errors=_STATE["save_errors"], signature_mismatches=_STATE["signature_mismatches"],
                index_repairs=_STATE["index_repairs"],
                error_types=list(_STATE["error_types"]), root_conflicts=_STATE["root_conflicts"],
                not_cached_note="kernels built by factories (certificate_prescreen._kernel, bernstein_barrier) "
                                "compile as before")
