"""Explicit extension schedule, independent of the completed original schedule."""
import math


def extension_lr_factor(step, duration):
    if duration <= 0:
        raise ValueError("Extension duration must be positive")
    step = max(0, min(step, duration))
    warmup = max(1, int(duration * 0.01))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / (duration - warmup) if duration > warmup else 1.0
    return 0.5 * (1.0 + math.cos(math.pi * progress))
