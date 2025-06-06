"""Pool de procesos compartido para la evaluación paralela."""
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import atexit

_pool: Optional[ProcessPoolExecutor] = None

def get_pool(n_jobs: int) -> ProcessPoolExecutor:
    """Devuelve un ``ProcessPoolExecutor`` global con ``n_jobs`` trabajadores."""
    global _pool
    if n_jobs <= 1:
        raise ValueError("n_jobs must be greater than 1 for a pool")
    if _pool is None or _pool._max_workers != n_jobs:
        shutdown_pool()
        _pool = ProcessPoolExecutor(max_workers=n_jobs)
    return _pool


def shutdown_pool() -> None:
    """Cierra el pool de procesos global si existe."""
    global _pool
    if _pool is not None:
        _pool.shutdown()
        _pool = None

atexit.register(shutdown_pool)
