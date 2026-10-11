"""Nonblocking, single-flight SAM recovery; no CUDA or camera dependencies."""
import math

from sam_initialization import run_sam_attempt


def fit_is_valid(fit):
    residual = fit.get('resid_med_m')
    coverage = fit.get('valid_frac', 0)
    return (residual is not None and math.isfinite(residual)
            and math.isfinite(coverage) and coverage >= .3 and residual <= .02
            and fit.get('sil_px', 1) > 0)


class RecoverySearch:
    """Keep tracking while one immutable RGB/depth bundle is segmented.

    Invalidated jobs are drained, never reused or run concurrently with another
    SAM call. A brief bad frame does not initiate global registration.
    """
    def __init__(self, lost_frames=3, retry_s=1.0, segment=None):
        if lost_frames < 1 or not math.isfinite(retry_s) or retry_s < 0:
            raise ValueError('Recovery requires lost_frames >= 1 and retry_s >= 0')
        self.lost_frames, self.retry_s = lost_frames, retry_s
        self.bad = self.good = self.attempts = 0
        self.future = self.bundle = None
        self.generation = 0
        self.job_generation = 0
        self.next_attempt = 0.0
        self.last_error = None
        self.completed = 0
        self.segment = segment

    def reset(self):
        self.generation += 1
        self.bad = self.good = 0
        self.next_attempt = 0.0

    def update(self, valid, bundle, now, executor, text):
        if valid:
            self.good += 1
            self.bad = 0
            if self.good == 2:
                self.generation += 1
        else:
            self.bad += 1
            self.good = 0
        candidate = None
        drained = False
        if self.future is not None and self.future.done():
            drained = True
            result = self.future.result()
            self.completed += 1
            self.last_error = result.error
            if self.job_generation == self.generation and not valid:
                if result.mask is not None:
                    candidate = (self.bundle, result)
            self.future = self.bundle = None
            self.next_attempt = now + self.retry_s
        if (not drained and candidate is None and self.future is None and not valid
                and self.bad >= self.lost_frames and now >= self.next_attempt):
            self.bundle = bundle  # inference creates new arrays, never overwrites this bundle
            self.job_generation = self.generation
            self.future = executor.submit(self.segment or run_sam_attempt, bundle['rect'], text)
            self.attempts += 1
        return candidate
