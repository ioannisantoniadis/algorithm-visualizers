"""
data.py — signal generators for the FFT visualiser

All generators return plain NumPy arrays; nothing here touches the FFT
itself. Kept deliberately simple so the interesting behaviour lives in
algorithm.py.

Public API
----------
N_CHOICES                       — power-of-two sample counts offered in the UI
DEFAULT_FREQS / DEFAULT_AMPS / DEFAULT_PHASES
make_composite_signal(...)      — sum of user-chosen sinusoids (+ optional noise)
make_chirp(...)                 — linear chirp, for the spectrogram view
"""

from __future__ import annotations

import numpy as np

# Power-of-two lengths offered in the sidebar. The radix-2 Cooley-Tukey
# path *requires* N to be a power of two; this list is the single place
# that constraint is enforced for the UI.
N_CHOICES = [64, 128, 256, 512, 1024, 2048]

# Small N choices for the butterfly-diagram tab, where every stage is
# drawn as an explicit network — kept tiny so the diagram stays legible.
DIAGRAM_N_CHOICES = [8, 16, 32, 64]

DEFAULT_FREQS = [5.0, 12.0, 30.0]
DEFAULT_AMPS = [1.0, 0.6, 0.35]
DEFAULT_PHASES = [0.0, 0.0, 0.0]


def make_composite_signal(
    freqs: list[float],
    amps: list[float],
    phases: list[float],
    n: int,
    fs: float,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build x(t) = sum_k amp_k * sin(2*pi*freq_k*t + phase_k) + noise.

    Returns (t, x, components) where components[k] is the k-th sinusoid
    alone (useful for showing "what's hiding inside the sum").
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    components = [
        amp * np.sin(2 * np.pi * freq * t + phase)
        for freq, amp, phase in zip(freqs, amps, phases)
    ]
    x = np.sum(components, axis=0) if components else np.zeros(n)

    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=n)

    return t, x, components


def make_chirp(
    n: int,
    fs: float,
    f0: float = 5.0,
    f1: float = 120.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear chirp: instantaneous frequency sweeps linearly from f0 to f1.

    A chirp is the classic "why do I need a spectrogram, not just an FFT"
    example — a single FFT of the whole signal shows *that* frequencies
    between f0 and f1 are present, but only a spectrogram shows *when*.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    duration = n / fs
    k = (f1 - f0) / duration  # chirp rate (Hz/s)
    instantaneous_phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    x = np.sin(instantaneous_phase)

    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=n)

    return t, x
