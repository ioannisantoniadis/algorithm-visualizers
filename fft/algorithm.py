"""
algorithm.py — From-scratch radix-2 Cooley-Tukey FFT with full stage history

The Cooley-Tukey trick: an N-point DFT (N a power of two) can be built out of
two (N/2)-point DFTs plus N/2 "butterfly" combinations. Applied recursively
(or, as here, unrolled iteratively after a bit-reversal permutation) this
turns an O(N^2) computation into O(N log N).

The classic *decimation-in-time* (DIT) formulation used below:

  1. Reorder the input into bit-reversed order.
  2. For each of log2(N) stages, combine pairs of elements that are
     `span/2` apart using a complex "twiddle factor" — this is the
     butterfly operation:
         even' = even + W * odd
         odd'  = even - W * odd
     where W = exp(-2*pi*i*j/span).

Every stage is recorded as a Snapshot so the visualiser can replay the
butterfly diagram frame by frame. A vectorised (non-recording) fast path is
used for benchmarking against the naive O(N^2) DFT, so timing numbers
reflect the algorithm's asymptotic behaviour rather than Python-loop
overhead.

Only NumPy array arithmetic is used — no numpy.fft / scipy.fft anywhere in
this module. Those are permitted *only* as a correctness oracle in tests,
never in the algorithm itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Butterfly:
    """One elementary combine step: new[i] = a[i] + W*a[j], new[j] = a[i] - W*a[j]."""

    i: int
    j: int
    twiddle: complex


@dataclass
class Snapshot:
    """State of the working array at one point in the FFT computation.

    stage == -1  -> original input, natural order
    stage ==  0  -> input after the bit-reversal permutation
    stage ==  s  -> array after completing butterfly stage s (1..log2 N)
    """

    stage: int
    values: np.ndarray                       # complex128, shape (N,)
    butterflies: list[Butterfly] = field(default_factory=list)
    span: int = 0                             # block size 2**stage (0 for input/bit-rev)

    @property
    def phase(self) -> Literal["input", "bit-reversal", "stage"]:
        if self.stage == -1:
            return "input"
        if self.stage == 0:
            return "bit-reversal"
        return "stage"

    @property
    def title(self) -> str:
        if self.phase == "input":
            return "Input signal — natural order"
        if self.phase == "bit-reversal":
            return "Bit-reversal permutation"
        return f"Stage {self.stage} — {len(self.butterflies)} butterflies, span {self.span}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bit_reversal_indices(n_bits: int) -> np.ndarray:
    """Return the permutation rev such that rev[i] reverses the n_bits bits of i."""
    n = 1 << n_bits
    idx = np.arange(n, dtype=np.uint32)
    rev = np.zeros(n, dtype=np.uint32)
    for _ in range(n_bits):
        rev = (rev << 1) | (idx & 1)
        idx >>= 1
    return rev.astype(int)


def bit_reversal_permutation(n_bits: int) -> np.ndarray:
    """Public wrapper around _bit_reversal_indices, used by the visualiser
    to draw the permutation stage without duplicating the bit-twiddling."""
    return _bit_reversal_indices(n_bits)


def _check_power_of_two(n: int) -> int:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"N must be a power of two for the radix-2 FFT, got {n}")
    return n.bit_length() - 1


# ---------------------------------------------------------------------------
# Public API — recording FFT (used by the butterfly-diagram visualiser)
# ---------------------------------------------------------------------------

def fft_radix2(x: np.ndarray, record: bool = True) -> tuple[np.ndarray, list[Snapshot]]:
    """Iterative decimation-in-time radix-2 FFT.

    Parameters
    ----------
    x:      real or complex input, length must be a power of two
    record: if False, skip building the Snapshot list (fast path, still
            fully vectorised per stage — used for benchmarking)

    Returns
    -------
    (X, snapshots) where X is the length-N complex spectrum and snapshots
    is a chronological list of Snapshot objects (empty if record=False).
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    n_bits = _check_power_of_two(n)

    snapshots: list[Snapshot] = []
    if record:
        snapshots.append(Snapshot(stage=-1, values=x.copy()))

    rev = _bit_reversal_indices(n_bits)
    a = x[rev].copy()

    if record:
        snapshots.append(Snapshot(stage=0, values=a.copy()))

    for s in range(1, n_bits + 1):
        span = 1 << s
        half = span >> 1

        # Vectorised butterfly for this stage: view the array as
        # (n // span) independent blocks of size `span`, each split into
        # an "even" half and an "odd" half that combine via one twiddle
        # factor per row position — no per-butterfly Python loop needed.
        j = np.arange(half)
        twiddle = np.exp(-2j * np.pi * j / span)          # shape (half,)

        blocks = a.reshape(-1, span)                       # (n//span, span)
        even = blocks[:, :half]
        odd = blocks[:, half:]
        t = odd * twiddle                                  # broadcast over blocks
        new_blocks = np.concatenate([even + t, even - t], axis=1)
        a = new_blocks.reshape(-1)

        if record:
            # Reconstruct the explicit (i, j, twiddle) op list only when
            # someone actually wants to draw the diagram — this is O(N)
            # bookkeeping, not the hot path.
            block_starts = np.arange(0, n, span)
            idx_even = (block_starts[:, None] + j[None, :]).reshape(-1)
            idx_odd = idx_even + half
            tw_full = np.tile(twiddle, len(block_starts))
            ops = [
                Butterfly(int(ie), int(io), complex(w))
                for ie, io, w in zip(idx_even, idx_odd, tw_full)
            ]
            snapshots.append(Snapshot(stage=s, values=a.copy(), butterflies=ops, span=span))

    return a, snapshots


def ifft_radix2(X: np.ndarray) -> np.ndarray:
    """Inverse FFT built from the forward FFT via the standard conjugate trick:

        ifft(X) = conj(fft(conj(X))) / N

    This reuses fft_radix2 rather than re-deriving a second butterfly
    network, while remaining exact (no numpy.fft involved).
    """
    X = np.asarray(X, dtype=complex)
    n = len(X)
    y, _ = fft_radix2(np.conj(X), record=False)
    return np.conj(y) / n


# ---------------------------------------------------------------------------
# Baseline for comparison — the definition-of-DFT matrix multiply, O(N^2)
# ---------------------------------------------------------------------------

def naive_dft(x: np.ndarray) -> np.ndarray:
    """Direct evaluation of X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).

    This is the textbook definition, computed as a dense matrix-vector
    product. It is correct for any N (not just powers of two) but scales
    as O(N^2), which is exactly the point: it's the baseline the FFT beats.
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    k = np.arange(n).reshape(-1, 1)
    m = np.arange(n).reshape(1, -1)
    w = np.exp(-2j * np.pi * k * m / n)
    return w @ x


# ---------------------------------------------------------------------------
# Derived analyses used by the visualiser
# ---------------------------------------------------------------------------

def magnitude_spectrum(X: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, magnitude) for the first half of the spectrum (0..Nyquist).

    A real input signal has a spectrum that is symmetric about Nyquist, so
    the second half carries no extra information for display purposes.
    """
    n = len(X)
    half = n // 2 + 1
    freqs = np.arange(half) * fs / n
    mag = np.abs(X[:half]) / n * 2.0
    mag[0] /= 2.0  # DC term should not be doubled
    if n % 2 == 0:
        mag[-1] /= 2.0  # Nyquist term should not be doubled either
    return freqs, mag


def reconstruct_top_k(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the k largest-magnitude frequency bins and inverse-transform.

    Because x is real, its spectrum has conjugate-symmetric pairs; both
    bins of a pair are always kept or dropped together so the
    reconstruction stays real-valued. Returns (reconstructed_signal, kept_bin_indices).
    """
    n = len(x)
    X, _ = fft_radix2(x, record=False)
    mag = np.abs(X)

    # Rank distinct frequency magnitudes (bin 0 and, if N even, bin N/2 are
    # unpaired; everything else pairs bin m with bin N-m).
    half = n // 2 + 1
    order = np.argsort(mag[:half])[::-1]

    keep_mask = np.zeros(n, dtype=bool)
    kept = 0
    for m in order:
        if kept >= k:
            break
        if not keep_mask[m]:
            keep_mask[m] = True
            if m != 0 and not (n % 2 == 0 and m == n // 2):
                keep_mask[n - m] = True
            kept += 1

    X_top = np.where(keep_mask, X, 0.0)
    xk = ifft_radix2(X_top).real
    return xk, np.nonzero(keep_mask)[0]


# ---------------------------------------------------------------------------
# Short-time FFT — spectrogram building block
# ---------------------------------------------------------------------------

def _hann_window(size: int) -> np.ndarray:
    """From-scratch Hann window (tapers frame edges to reduce spectral leakage)."""
    n = np.arange(size)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1))


def spectrogram(
    x: np.ndarray,
    fs: float,
    window_size: int = 128,
    hop: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform: slide a windowed FFT across the signal.

    Returns (times, freqs, magnitude) where magnitude has shape
    (n_freq_bins, n_frames) — ready to feed straight into a heatmap.
    """
    _check_power_of_two(window_size)
    n = len(x)
    window = _hann_window(window_size)
    n_frames = max(1, 1 + (n - window_size) // hop)

    half = window_size // 2
    mags = np.empty((half, n_frames))
    for i in range(n_frames):
        start = i * hop
        seg = x[start : start + window_size]
        if len(seg) < window_size:
            seg = np.pad(seg, (0, window_size - len(seg)))
        X, _ = fft_radix2(seg * window, record=False)
        mags[:, i] = np.abs(X[:half])

    freqs = np.arange(half) * fs / window_size
    times = (np.arange(n_frames) * hop + window_size / 2) / fs
    return times, freqs, mags


# ---------------------------------------------------------------------------
# Benchmark — naive O(N^2) DFT vs O(N log N) FFT
# ---------------------------------------------------------------------------

def benchmark(sizes: list[int], repeats: int = 3, seed: int = 0) -> tuple[list[float], list[float]]:
    """Time naive_dft vs fft_radix2 (fast, non-recording path) across sizes.

    Returns (naive_times, fft_times), both in seconds, averaged over `repeats`.
    """
    import time

    rng = np.random.default_rng(seed)
    naive_times: list[float] = []
    fft_times: list[float] = []

    for n in sizes:
        x = rng.normal(size=n)

        t0 = time.perf_counter()
        for _ in range(repeats):
            naive_dft(x)
        naive_times.append((time.perf_counter() - t0) / repeats)

        t0 = time.perf_counter()
        for _ in range(repeats):
            fft_radix2(x, record=False)
        fft_times.append((time.perf_counter() - t0) / repeats)

    return naive_times, fft_times
