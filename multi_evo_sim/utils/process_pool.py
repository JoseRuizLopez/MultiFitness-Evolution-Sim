"""Shared process pool for parallel evaluation."""
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import atexit

_pool: Optional[ProcessPoolExecutor] = None

def get_pool(n_jobs: int) -> ProcessPoolExecutor:
    """Return a global ``ProcessPoolExecutor`` with ``n_jobs`` workers."""
    global _pool
    if n_jobs <= 1:
        raise ValueError("n_jobs must be greater than 1 for a pool")
    if _pool is None or _pool._max_workers != n_jobs:
        shutdown_pool()
        _pool = ProcessPoolExecutor(max_workers=n_jobs)
    return _pool


def shutdown_pool() -> None:
    """Shutdown the global process pool if it exists."""
    global _pool
    if _pool is not None:
        _pool.shutdown()
        _pool = None

atexit.register(shutdown_pool)
