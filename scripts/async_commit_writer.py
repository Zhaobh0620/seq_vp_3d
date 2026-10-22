"""E76 (explicit throughput-v9 option): ordered background file writes for the MRI worker.

The main thread serializes each checkpoint into memory and hashes those exact bytes, so the
recorded SHA-256 is the SHA-256 of the file that will exist; one writer thread then writes the
files strictly in submission order (temp file + atomic rename). JSON indexes that reference a
checkpoint (commits.json, levels.json) are queued behind it, so on disk an index never names a
file that is not yet complete. Errors are re-raised on the next submit or flush, never dropped.

E102 (explicit throughput-v20 option, `deferred_hash=True`): the serialization (np.savez into
memory) and the SHA-256 of those exact bytes move to the writer thread, which still writes the
same bytes; `npz_deferred` returns a PendingDigest at once. An array is handed over only if it
cannot change (it and every ndarray base are read-only and the chain ends in an owned buffer);
any other array is copied first with the layout np.savez would record. A JSON index is still
rendered on the main thread when it is submitted (same content as before); a digest that is not
known yet is rendered as a placeholder string literal, whose occurrences are counted, and the
writer replaces each by the digest's JSON string before writing -- by then the checkpoint job
that owns it has finished, because jobs run in submission order. str() of a PendingDigest waits
for it, so every JSON writer (default=str) renders the same hex string.
"""
import hashlib
import io
import itertools
import json
import os
import queue
import threading
from pathlib import Path
from time import perf_counter

import numpy as np


def _json_text(value):
    # Same serialization as scripts.run_stage_3d_c.dump (identical file contents).
    return json.dumps(value, indent=2, allow_nan=False,
                      default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))+"\n"


class PendingDigest:
    """E102: the SHA-256 (hex) of a checkpoint that the ordered writer serializes and hashes later."""
    __slots__ = ('path', 'serial', '_done', '_value', '_error', '_passthrough')

    def __init__(self, path, serial, passthrough=()):
        self.path, self.serial = str(path), serial
        self._done, self._value, self._error = threading.Event(), None, None
        self._passthrough = tuple(passthrough)       # v39: the writer's pass-through error types

    def _set(self, value=None, error=None):
        self._value, self._error = value, error
        self._done.set()

    def done(self):
        return self._done.is_set()

    def result(self):
        self._done.wait()
        if self._error is not None:
            if self._passthrough and isinstance(self._error, self._passthrough):
                raise self._error                     # v39: e.g. the disk guard's own error, wherever the digest is read
            raise RuntimeError('ordered checkpoint writer failed: '+self.path) from self._error
        return self._value

    def __str__(self):
        return self.result()

    def __repr__(self):
        return 'PendingDigest(%r, %s)' % (self.path, 'done' if self.done() else 'pending')

    def placeholder(self):
        # A JSON string literal no recorded value contains (NUL is escaped as \u0000); occurrences are counted anyway.
        return json.dumps('\x00jd-curl-pending-sha256-%d\x00' % self.serial)


def _immutable(value):
    # The rule of jd_curl.three_d.problem (E76 digest memo): read-only through every ndarray base, owned buffer.
    a = value
    while isinstance(a, np.ndarray):
        if a.flags.writeable:
            return False
        a = a.base
    return a is None


def _plain(value):
    return type(value) is np.ndarray and not value.dtype.hasobject


def _frozen(value):
    """`value` itself if it cannot change; otherwise a read-only copy that np.savez stores with the same header and
    data bytes (Fortran order only when np.savez would record it: F-contiguous and not C-contiguous)."""
    if not _plain(value):
        raise TypeError('deferred checkpoints take plain numeric ndarrays')
    if _immutable(value):
        return value, False
    order = 'F' if value.flags.f_contiguous and not value.flags.c_contiguous else 'C'
    copy = np.array(value, copy=True, order=order)
    copy.flags.writeable = False
    return copy, True


def resolve_pending(value):
    """E102: replace every PendingDigest inside nested dicts/lists by its hex string, in place (after a flush).
    Returns the number replaced."""
    count = 0
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return 0
    for key, item in list(items):
        if isinstance(item, PendingDigest):
            value[key] = item.result()
            count += 1
        else:
            count += resolve_pending(item)
    return count


_SERIALS = itertools.count()    # placeholders unique across writers (review E102)


class OrderedWriter:
    def __init__(self, max_pending=2, *, deferred_hash=False, before_npz=None, passthrough=()):
        if not isinstance(deferred_hash, bool):
            raise TypeError('deferred_hash must be boolean')
        if before_npz is not None and (not callable(before_npz) or not deferred_hash):
            raise TypeError('before_npz must be callable and needs deferred_hash=True (checkpoint jobs of kind npz)')
        # v39 (guard_on_writer): called on the writer thread right before each checkpoint job is packed and written, as
        # before_npz(path, size, queued_bytes) with queued_bytes = this job and every job still queued behind it; an
        # exception stops all later writes and is re-raised on the next submit/flush (as is for `passthrough` types)
        self._before_npz = before_npz
        self._passthrough = tuple(passthrough)
        self._queue = queue.Queue(maxsize=max_pending)
        self._error = None
        self.jobs = 0
        self.writer_seconds = 0.
        self.bytes = 0
        self.max_pending_seen = 0
        self.queued_bytes = 0                  # payload bytes submitted but not yet written
        self.deferred_hash = deferred_hash     # E102 (throughput-v20)
        self.deferred = dict(checkpoints=0, arrays_shared=0, arrays_copied=0, synchronous=0, pack_seconds=0.,
                             hash_seconds=0., placeholders=0, indexes_with_placeholders=0, placeholder_fallbacks=0)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name='jd-curl-ordered-writer', daemon=True)
        self._thread.start()

    def _write(self, path, payload):
        tick = perf_counter()
        temp = Path(str(path)+'.partial')
        with open(temp, 'wb') as stream:
            stream.write(payload)
        os.replace(temp, path)
        self.writer_seconds += perf_counter()-tick
        self.bytes += len(payload)

    def _pack(self, path, arrays, pending):
        # E102: the same np.savez bytes as OrderedWriter.npz, hashed without a copy, then written.
        tick = perf_counter()
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        view = buffer.getbuffer()
        try:
            middle = perf_counter()
            digest = hashlib.sha256(view).hexdigest()
            done = perf_counter()
            self._write(path, view)
        finally:
            view.release()
        self.deferred['pack_seconds'] += middle-tick
        self.deferred['hash_seconds'] += done-middle
        pending._set(digest)

    def _fill(self, path, text, slots):
        # E102: every digest named here belongs to an earlier job, finished before this one started.
        for literal, pending in slots:
            text = text.replace(literal, json.dumps(pending.result()))
        self._write(path, text.encode())

    def _run(self):
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            kind, path, payload, extra, size = job
            try:
                if self._error is None:
                    if kind == 'bytes':
                        self._write(path, payload)
                    elif kind == 'npz':
                        if self._before_npz is not None:
                            self._before_npz(path, size, self.queued_bytes)     # v39: the disk guard, before this write
                        self._pack(path, payload, extra)
                    else:
                        self._fill(path, payload, extra)
                elif kind == 'npz':
                    extra._set(error=self._error)        # never leave a waiting result() blocked
            except BaseException as error:      # surfaced on the next submit/flush
                self._error = error
                if kind == 'npz' and not extra.done():
                    extra._set(error=error)
            finally:
                with self._lock:
                    self.queued_bytes -= size
                self._queue.task_done()

    def _raise(self):
        if self._error is not None:
            if self._passthrough and isinstance(self._error, self._passthrough):
                raise self._error                  # v39: e.g. the disk guard's own error, so callers classify it as before
            raise RuntimeError('ordered checkpoint writer failed') from self._error

    def _enqueue(self, kind, path, payload, extra, size):
        self._raise()
        self.jobs += 1
        with self._lock:
            self.queued_bytes += size
        self._queue.put((kind, Path(path), payload, extra, size))
        self.max_pending_seen = max(self.max_pending_seen, self._queue.qsize())

    def submit(self, path, payload):
        payload = bytes(payload) if not isinstance(payload, bytes) else payload
        self._enqueue('bytes', path, payload, None, len(payload))

    def npz(self, path, **arrays):
        """Serialize now, hash these bytes, write later. Returns the SHA-256 of the file bytes."""
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        self.submit(path, data)
        return digest

    def npz_deferred(self, path, **arrays):
        """E102: serialize, hash and write later, in order. Returns a PendingDigest of the file bytes."""
        if not self.deferred_hash:
            raise RuntimeError('npz_deferred requires OrderedWriter(deferred_hash=True)')
        if not all(_plain(value) for value in arrays.values()):
            # anything but a plain numeric ndarray: serialized and hashed here, as E76 does (a str digest)
            digest = self.npz(path, **arrays)
            self.deferred['synchronous'] += 1
            return digest
        held, copied = {}, 0
        for name, value in arrays.items():
            held[name], fresh = _frozen(value)
            copied += fresh
        pending = PendingDigest(path, next(_SERIALS), self._passthrough)
        # the stored size estimate of the disk guard (scripts.run_stage_3d_e19_mri.worker.save)
        size = sum(a.nbytes for a in held.values())+65536*len(held)+1024**2
        self._enqueue('npz', path, held, pending, size)
        self.deferred['checkpoints'] += 1
        self.deferred['arrays_copied'] += copied
        self.deferred['arrays_shared'] += len(held)-copied
        return pending

    def _render(self, value):
        """E102: the index text with placeholders for unknown digests, and the (literal, digest) pairs to fill."""
        slots, counts = {}, {}

        def default(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, PendingDigest):
                if obj.done():
                    return obj.result()
                literal = obj.placeholder()
                slots[literal] = obj
                counts[literal] = counts.get(literal, 0)+1
                return json.loads(literal)
            return str(obj)
        text = json.dumps(value, indent=2, allow_nan=False, default=default)+"\n"
        if any(text.count(literal) != n for literal, n in counts.items()):
            # a recorded string equal to a placeholder: render with the digests themselves (waits for them)
            self.deferred['placeholder_fallbacks'] += 1
            return _json_text(value), []
        return text, list(slots.items())

    def json(self, path, value):
        if self._passthrough:
            self._raise()                     # v39 (review): surface a failed check before rendering an index
        if not self.deferred_hash:
            self.submit(path, _json_text(value).encode())
            return
        text, slots = self._render(value)
        if not slots:
            self.submit(path, text.encode())
            return
        self.deferred['placeholders'] += len(slots)
        self.deferred['indexes_with_placeholders'] += 1
        self._enqueue('fill', path, text, slots, len(text))

    def flush(self):
        self._queue.join()
        self._raise()

    def raise_if_failed(self):
        """v39: raise the first writer error now (the pass-through types as they are), without waiting."""
        self._raise()

    def close(self):
        try:
            self.flush()
        finally:              # review E102: the writer thread also ends when the flush raises
            self._queue.put(None)
            self._thread.join()

    def stats(self):
        result = dict(jobs=self.jobs, writer_seconds=self.writer_seconds, bytes=self.bytes,
                      max_pending_seen=self.max_pending_seen,
                      scope='E76 ordered background writes; main-thread time is in stored_array_io_seconds')
        if self.deferred_hash:
            result['deferred_hash'] = dict(self.deferred, policy='writer-thread-pack-and-hash-v1',
                scope='E102: np.savez and SHA-256 on the writer thread (pack/hash seconds there); '
                      'writer_seconds is the file writes only')
        return result
