"""SAM outcomes for asynchronous initialization, without silent manual fallback."""
from dataclasses import dataclass
import time


@dataclass
class SamResult:
    mask: object = None
    box: object = None
    score: object = None
    error: str | None = None
    elapsed_s: float = 0.0


def run_sam_attempt(rgb, text, backend=None):
    started = time.perf_counter()
    try:
        if backend is None:
            from sam_mask import sam_mask
            backend = sam_mask
        mask, box, score = backend(rgb, text)
        if mask is None or not mask.any():
            return SamResult(elapsed_s=time.perf_counter()-started)
        return SamResult(mask, box, score, elapsed_s=time.perf_counter()-started)
    except Exception as exc:
        return SamResult(error=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter()-started)
