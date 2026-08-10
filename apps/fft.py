"""FFT page — adapted from the standalone fft-viz repo for the multipage
portfolio app. Session-state keys are namespaced with a per-page prefix
since st.session_state is shared across all pages. This page has four tabs
(Signal & Spectrum, Butterfly diagram, Spectrogram, FFT vs naive DFT), each
with its own state — every key below is namespaced through _k() so this
page's widgets and state never collide with other pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from fft.algorithm import (
    benchmark,
    bit_reversal_permutation,
    fft_radix2,
    magnitude_spectrum,
    reconstruct_top_k,
    spectrogram,
)
from fft.data import DEFAULT_AMPS, DEFAULT_FREQS, DIAGRAM_N_CHOICES, N_CHOICES, make_chirp, make_composite_signal
from fft.visualize import (
    butterfly_edges,
    make_benchmark_figure,
    make_butterfly_figure,
    make_reconstruction_figure,
    make_signal_figure,
    make_spectrogram_figure,
    make_spectrum_figure,
)

NS = "fft"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Sidebar — signal composition (drives the "Signal & Spectrum" tab and
# doubles as the source waveform for the spectrum-recovery demo)
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ⚙️ SIGNAL")

with st.sidebar.container(border=True):
    n = st.selectbox(
        "N (samples)", options=N_CHOICES, index=N_CHOICES.index(256),
        help="Must be a power of two — the radix-2 Cooley-Tukey FFT only "
             "works when N splits evenly in half, log2(N) times over.",
        key=_k("n"),
    )
    fs = float(n)  # sample rate chosen equal to N -> exactly one second of signal
    nyquist = fs / 2.0

    n_components = st.slider("Number of sinusoids", min_value=1, max_value=4, value=3, key=_k("n_components"))

    # Widget state lives directly under the freq_i / amp_i keys the sliders
    # use, so both "N changed -> re-clip to Nyquist" and "Randomize" can write
    # straight into session_state and have the sliders pick it up on the next
    # render.
    _base_freqs = (DEFAULT_FREQS + [40.0, 55.0])
    _base_amps = (DEFAULT_AMPS + [0.25, 0.2])
    for i in range(n_components):
        if _k(f"freq_{i}") not in st.session_state:
            st.session_state[_k(f"freq_{i}")] = min(_base_freqs[i], nyquist - 1)
        else:
            st.session_state[_k(f"freq_{i}")] = min(st.session_state[_k(f"freq_{i}")], nyquist - 1)
        st.session_state.setdefault(_k(f"amp_{i}"), _base_amps[i])

    regenerate = st.button("🔀 Randomize frequencies", use_container_width=True, key=_k("regenerate"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

    if regenerate:
        rng = np.random.default_rng(int(seed))
        new_freqs = sorted(rng.uniform(2, nyquist * 0.9, size=n_components).round(1).tolist())
        new_amps = rng.uniform(0.3, 1.5, size=n_components).round(2).tolist()
        for i in range(n_components):
            st.session_state[_k(f"freq_{i}")] = new_freqs[i]
            st.session_state[_k(f"amp_{i}")] = new_amps[i]
        st.rerun()

    st.caption(f"Nyquist frequency at this N: **{nyquist:.0f} Hz**")

    freqs, amps = [], []
    for i in range(n_components):
        c1, c2 = st.columns(2)
        f = c1.slider(f"f{i + 1} (Hz)", min_value=1.0, max_value=float(nyquist - 1),
                       step=0.5, key=_k(f"freq_{i}"))
        a = c2.slider(f"amp{i + 1}", min_value=0.1, max_value=2.0,
                       step=0.05, key=_k(f"amp_{i}"))
        freqs.append(f)
        amps.append(a)
    phases = [0.0] * n_components

    noise_std = st.slider("Noise (std dev)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, key=_k("noise_std"))

with st.sidebar.expander("📖 Cooley-Tukey in a nutshell", expanded=False):
    st.markdown(
        """
1. Bit-reverse the input order
2. Combine pairs `span` apart with a twiddle factor `W`
   (a *butterfly*): `a±W·b`
3. Double the span, repeat — `log2(N)` stages total
4. Result: the full spectrum in `O(N log N)` time
"""
    )

t, x, components = make_composite_signal(freqs, amps, phases, n=n, fs=fs, noise_std=noise_std, seed=int(seed))
X, _ = fft_radix2(x, record=False)
spec_freqs, spec_mag = magnitude_spectrum(X, fs)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Fast Fourier Transform — Step-by-step Visualiser")
st.caption(
    f"N = **{n}** | components = **{n_components}** | "
    f"freqs = **{[round(f, 1) for f in freqs]} Hz** | noise σ = **{noise_std}**"
)

tab_spectrum, tab_butterfly, tab_spectrogram, tab_benchmark = st.tabs(
    ["📈 Signal & Spectrum", "🦋 Butterfly diagram", "🎵 Spectrogram", "⏱️ FFT vs naive DFT"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Signal & spectrum + top-k reconstruction ("DFT recovering" demo)
# ---------------------------------------------------------------------------
with tab_spectrum:
    with st.container(border=True):
        st.plotly_chart(make_signal_figure(t, x, components), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("chart_signal"))
    with st.container(border=True):
        st.plotly_chart(make_spectrum_figure(spec_freqs, spec_mag, true_freqs=freqs),
                         use_container_width=True, config={"displayModeBar": False}, key=_k("chart_spectrum"))

    st.markdown("#### Recovering the signal from a handful of frequencies")
    st.caption(
        "The FFT turns the signal into frequency-domain coefficients. Keep only the "
        "k largest-magnitude frequencies (each contributes a conjugate-symmetric pair "
        "of bins so the reconstruction stays real) and inverse-transform: once k reaches "
        "the number of true sinusoids, the reconstruction locks onto the original almost exactly."
    )

    k_key, playing_k_key = _k("topk_value"), _k("playing_k")
    max_k = len(spec_freqs)
    st.session_state.setdefault(k_key, 1)
    st.session_state.setdefault(playing_k_key, False)
    st.session_state[k_key] = min(st.session_state[k_key], max_k)

    # Auto-advance must happen *before* the toggle/slider widgets below are
    # instantiated: st.session_state[key] cannot be written once a widget
    # bound to that key exists in this run (that raises a StreamlitAPIException).
    # Doing the increment here, then st.rerun()-ing into a fresh script
    # execution, keeps every write strictly ahead of its widget's creation.
    if st.session_state[playing_k_key]:
        if st.session_state[k_key] < max_k:
            time.sleep(0.35)
            st.session_state[k_key] += 1
            st.rerun()
        else:
            st.session_state[playing_k_key] = False
            st.rerun()

    with st.container(border=True):
        col_play, col_slider = st.columns([1, 5])
        with col_play:
            playing_k = st.toggle("▶ Auto-increase k", key=playing_k_key)
        with col_slider:
            k = st.slider("k — number of frequency components kept", min_value=1, max_value=max_k, key=k_key)

    xk, kept_bins = reconstruct_top_k(x, k)
    with st.container(border=True):
        st.plotly_chart(make_reconstruction_figure(t, x, xk, k, max_k), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("chart_reconstruction"))

    with st.expander("📖 Reading this chart"):
        st.markdown(
            "- Grey = original signal, red = reconstruction using only the top **k** "
            "frequency components (by magnitude)\n"
            "- Once `k` reaches the number of true sinusoids in the sidebar, the "
            "reconstruction should match almost exactly, even in the presence of noise\n"
            "- Extra k beyond that mostly picks up noise, not new signal structure"
        )

# ---------------------------------------------------------------------------
# TAB 2 — Butterfly diagram (the core algorithm visualiser)
# ---------------------------------------------------------------------------
with tab_butterfly:
    st.markdown(
        "A separate, small signal is used here so every butterfly in the network "
        "can be drawn and inspected individually."
    )

    diag_n = st.selectbox("Diagram size (N)", options=DIAGRAM_N_CHOICES,
                           index=DIAGRAM_N_CHOICES.index(16), key=_k("diag_n"))
    diag_n_bits = int(np.log2(diag_n))

    # Reuse the same frequency mix, resampled onto the small diagram grid.
    diag_freqs = [min(f, diag_n / 2 - 1) for f in freqs[:2]] or [3.0]
    diag_amps = amps[: len(diag_freqs)] or [1.0]
    _, diag_x, _ = make_composite_signal(diag_freqs, diag_amps, [0.0] * len(diag_freqs),
                                          n=diag_n, fs=float(diag_n), noise_std=0.0, seed=int(seed))

    _, snapshots = fft_radix2(diag_x, record=True)
    rev = bit_reversal_permutation(diag_n_bits)
    edges = butterfly_edges(diag_n_bits, rev)

    n_steps = len(snapshots)
    step_key, playing_key = _k("bf_step"), _k("bf_playing")
    st.session_state.setdefault(step_key, 0)
    st.session_state.setdefault(playing_key, False)
    step_idx = min(st.session_state[step_key], n_steps - 1)
    playing = st.session_state[playing_key]

    with st.container(border=True):
        col_prev, col_play, col_pause, col_next, col_reset = st.columns([1, 1.2, 1.2, 1, 1.2])
        with col_prev:
            if st.button("◀ Prev", use_container_width=True, disabled=(step_idx == 0 or playing), key=_k("bf_prev")):
                st.session_state[step_key] = step_idx - 1
                st.rerun()
        with col_play:
            if st.button("▶  Play", use_container_width=True,
                          disabled=(playing or step_idx == n_steps - 1), type="primary", key=_k("bf_play")):
                st.session_state[playing_key] = True
                st.rerun()
        with col_pause:
            if st.button("⏸  Pause", use_container_width=True, disabled=not playing, key=_k("bf_pause")):
                st.session_state[playing_key] = False
                st.rerun()
        with col_next:
            if st.button("Next ▶", use_container_width=True,
                          disabled=(step_idx == n_steps - 1 or playing), key=_k("bf_next")):
                st.session_state[step_key] = step_idx + 1
                st.rerun()
        with col_reset:
            if st.button("⏮ Reset", use_container_width=True, disabled=playing, key=_k("bf_reset")):
                st.session_state[step_key] = 0
                st.rerun()

        st.progress(step_idx / max(n_steps - 1, 1),
                    text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}")

    snap = snapshots[step_idx]
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Phase", snap.phase.replace("-", " ").capitalize())
        m2.metric("Span", str(snap.span) if snap.span else "—")
        m3.metric("Butterflies this stage", str(len(snap.butterflies)))

    with st.container(border=True):
        st.plotly_chart(make_butterfly_figure(snap, edges, diag_n_bits), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("chart_butterfly"))

    if snap.phase == "input":
        st.info("**Natural order** — the raw samples, indexed 0..N-1 as they arrived.")
    elif snap.phase == "bit-reversal":
        st.info(
            "**Bit-reversal permutation** — reorder the input so that index `i` moves to "
            "the position obtained by reversing the bits of `i`. This reordering is what "
            "makes the iterative butterfly stages line up correctly — no data changes yet, "
            "only positions."
        )
    else:
        st.info(
            f"**Stage {snap.stage}** — {len(snap.butterflies)} butterflies of "
            f"span {snap.span} run in parallel: `a' = a + W·b`, `b' = a - W·b`. Red edges show "
            "which pairs from the previous column feed into this one."
        )

    with st.expander("📖 Reading the diagram"):
        st.markdown(
            "- Columns = stages: natural order → bit-reversed → stage 1 → ... → final spectrum\n"
            "- Faint grey lines = the full butterfly network for this N (always visible for context)\n"
            "- **Red lines** = the butterflies that produced the current column from the previous one\n"
            "- Marker colour = real part, marker size = magnitude — hover a node for exact values\n"
            "- ◀ Prev / Next ▶ to step manually, ▶ Play to auto-advance through every stage"
        )

    if playing:
        if step_idx < n_steps - 1:
            time.sleep(0.9)
            st.session_state[step_key] = step_idx + 1
            st.rerun()
        else:
            st.session_state[playing_key] = False
            st.rerun()

# ---------------------------------------------------------------------------
# TAB 3 — Spectrogram (chirp: same FFT, applied in short sliding windows)
# ---------------------------------------------------------------------------
with tab_spectrogram:
    st.markdown(
        "A single FFT of a whole chirp tells you *which* frequencies are present but not "
        "*when*. Slide a short window across the signal and FFT each window separately — "
        "a **spectrogram** — to recover the time axis."
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        f0 = c1.slider("Start frequency (Hz)", 1.0, fs / 2 - 1, 5.0, key=_k("f0"))
        f1 = c2.slider("End frequency (Hz)", 1.0, fs / 2 - 1, min(fs / 2 - 1, 100.0), key=_k("f1"))

        # A window longer than the signal itself (N) gets zero-padded down to
        # a single, mostly-empty frame — the spectrogram collapses to one
        # flat horizontal band with no time axis at all, defeating the entire
        # point of this tab. Cap the offered window sizes at N so every
        # option on screen still produces a real, multi-frame spectrogram.
        _all_window_sizes = [32, 64, 128, 256]
        window_choices = [w for w in _all_window_sizes if w <= n] or [_all_window_sizes[0]]
        default_window = 64 if 64 in window_choices else window_choices[-1]
        window_size = c3.selectbox(
            "Window size", options=window_choices, index=window_choices.index(default_window),
            help="Capped at N (samples) from the sidebar — a window longer than the "
                 "signal itself would need heavy zero-padding and collapse the "
                 "spectrogram into a single, uninformative frame.",
            key=_k("window_size"),
        )

    t_chirp, x_chirp = make_chirp(n=n, fs=fs, f0=f0, f1=f1, noise_std=noise_std * 0.5, seed=int(seed))
    hop = max(1, window_size // 4)

    with st.container(border=True):
        st.plotly_chart(make_signal_figure(t_chirp, x_chirp, title="Chirp signal (time domain)"),
                         use_container_width=True, config={"displayModeBar": False}, key=_k("chart_chirp_signal"))

    times_s, freqs_s, mags_s = spectrogram(x_chirp, fs=fs, window_size=window_size, hop=hop)
    with st.container(border=True):
        st.plotly_chart(make_spectrogram_figure(times_s, freqs_s, mags_s), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("chart_spectrogram"))

    with st.expander("📖 Reading the spectrogram"):
        st.markdown(
            "- x-axis = time, y-axis = frequency, colour = magnitude at that time/frequency\n"
            "- The diagonal streak is the chirp's instantaneous frequency rising from "
            f"**{f0:.0f} Hz** to **{f1:.0f} Hz**\n"
            "- Smaller windows resolve time better but frequency worse, and vice versa — "
            "this trade-off is fundamental (an uncertainty principle for signals), not a "
            "limitation of this implementation"
        )

# ---------------------------------------------------------------------------
# TAB 4 — naive O(N^2) DFT vs O(N log N) FFT
# ---------------------------------------------------------------------------
with tab_benchmark:
    st.markdown(
        "Both algorithms compute *exactly* the same result — a definition-of-DFT matrix "
        "multiply versus the Cooley-Tukey FFT. Only the runtime scaling differs."
    )

    with st.container(border=True):
        max_pow = st.slider("Largest N to test (as 2^p)", min_value=6, max_value=12, value=10, key=_k("max_pow"))
        sizes = [2 ** p for p in range(4, max_pow + 1)]
        run = st.button("▶ Run benchmark", type="primary", key=_k("run_benchmark"))

    cache_key = _k("bench_result")
    cached = st.session_state.get(cache_key)
    # Re-run automatically whenever `sizes` no longer matches what's cached
    # (i.e. the "Largest N" slider moved) — otherwise the chart/metric below
    # would keep showing results for the *previous* slider value with no
    # indication anything is stale, even though nothing on screen looks wrong.
    if run or cached is None or cached[0] != sizes:
        with st.spinner("Timing naive DFT and FFT across sizes..."):
            naive_times, fft_times = benchmark(sizes, repeats=3, seed=int(seed))
        st.session_state[cache_key] = (sizes, naive_times, fft_times)

    sizes_c, naive_times, fft_times = st.session_state[cache_key]
    with st.container(border=True):
        st.plotly_chart(make_benchmark_figure(sizes_c, naive_times, fft_times), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("chart_benchmark"))

        speedup = naive_times[-1] / fft_times[-1] if fft_times[-1] > 0 else float("inf")
        st.metric(f"Speed-up at N={sizes_c[-1]}", f"{speedup:,.0f}×")

    with st.expander("📖 Why the gap widens"):
        st.markdown(
            "- Naive DFT: `O(N²)` — every output bin sums over all N input samples\n"
            "- Cooley-Tukey FFT: `O(N log N)` — `log2(N)` stages, each doing `O(N)` work\n"
            "- Doubling N roughly quadruples the naive DFT's cost but barely more than "
            "doubles the FFT's — that gap is why the FFT made real-time signal processing "
            "(audio, radar, medical imaging, wireless) practical"
        )
