"""Opt-in cooperative global deadline; never an acceptance/convergence rule.

The session is local to the calling context. Parallel CPU tiles are bounded
and checked by their ordered consumer, not by mutating worker threads.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


class WorkBudgetExceeded(RuntimeError):
    def __init__(self, phase):
        self.phase = phase
        super().__init__("global work deadline during "+phase)


@dataclass
class _Session:
    budget: object
    observer: object
    stack: list = field(default_factory=list)
    last_progress: float = float("-inf")


_active = ContextVar("jd_curl_work_session", default=None)


@contextmanager
def work_session(budget, observer=None):
    """Leave the scope before exporting the last formal state on cancellation."""
    token = _active.set(_Session(budget, observer))
    try:
        yield
    finally:
        _active.reset(token)


def emit(event, **values):
    session = _active.get()
    if session is not None and session.observer is not None:
        session.observer(dict(event=event, operations=list(session.stack), **values))


def checkpoint(phase, **progress):
    session = _active.get()
    if session is None:
        return
    if session.budget():
        emit("work_budget_interrupted", phase=phase, **progress)
        raise WorkBudgetExceeded(phase)
    now = perf_counter()
    if now-session.last_progress >= 1.:
        session.last_progress = now
        emit("work_progress", phase=phase, **progress)


@contextmanager
def operation(name):
    session = _active.get()
    if session is None:
        yield
        return
    checkpoint(name)
    started = perf_counter()
    session.stack.append(name)
    emit("operation_started", phase=name)
    try:
        yield
    except BaseException as error:
        # Record, but never turn arithmetic/IO/program errors into a deadline.
        emit("operation_aborted", phase=name, error_type=type(error).__name__,
             seconds=perf_counter()-started)
        raise
    else:
        emit("operation_finished", phase=name, seconds=perf_counter()-started)
        # Do not override a completed function's InvalidTrial/error status with
        # an exit-time deadline. The next block / pre-commit solver check is
        # authoritative; unfinished work never escapes its checkpoint.
    finally:
        session.stack.pop()
