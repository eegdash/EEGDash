# Plan: Wavelet Feature Extraction for EEGDash

## Context

The `eegdash` feature bank uses a DAG-based computation graph via `@feature_predecessor` decorators. Each domain (spectral, complexity, signal) lives in its own module. The user wants to add wavelet transforms, choosing between embedding them inside the spectral module (Option 1) or creating a standalone module (Option 2).

Wavelets are fundamentally different from Welch's PSD: their output is a time-frequency matrix `(..., n_scales, n_times)` per channel vs. a simple power vector `(..., n_freqs)`. This shape difference is the central architectural question.

---

## Signal Processing Background: Why Wavelets ≠ Spectral

| Dimension | Welch PSD | CWT (Continuous Wavelet) | DWT (Discrete Wavelet) |
|-----------|-----------|--------------------------|------------------------|
| Time resolution | None (global) | Yes (per-time) | Yes (scale-dependent) |
| Frequency resolution | Uniform | Logarithmic (high res at low f) | Octave-band |
| Output shape per channel | `(n_freqs,)` | `(n_scales, n_times)` | `[list of arrays]` |
| Non-stationarity handling | Poor | Excellent | Excellent |
| Computational cost | Low | Medium-High | Low |
| Library | `scipy.signal.welch` | `pywt.cwt` or `scipy.signal.cwt` | `pywt.wavedec` |

**Key insight**: if we collapse the time axis of a CWT (take mean squared amplitude per scale), we get `(n_scales,)` — which is shape-compatible with spectral `(n_freqs,)`. This is the architectural hinge between the two options.

---

## Option 1: `wavelet_preprocessor` inside `spectral.py`

**Idea**: Add a `wavelet_preprocessor` that collapses time and returns `(freqs, P)` — same signature as `spectral_preprocessor`. Wavelet features can then optionally accept both preprocessors via multi-predecessor.

### Computation graph

```
Raw Signal (x)
    ├─→ spectral_preprocessor (f, p)           [existing]
    │       └─→ spectral_entropy, spectral_bands_power, ...
    │
    └─→ wavelet_preprocessor (f, P)            [new, same shape]
            ├─→ spectral_bands_power           [reuse via multi-predecessor]
            ├─→ wavelet_entropy                [new]
            └─→ wavelet_relative_power         [new]
```

### Implementation sketch

```python
@feature_predecessor()
@utils.spectral_kwargs  # Reuse existing decorator for f_min, f_max, fs
def wavelet_preprocessor(x, /, *, _metadata, f_min, f_max, wavelet="morl", **kwargs):
    """CWT-based scalogram, averaged over time → shape matches spectral_preprocessor."""
    import pywt

    freqs_target = np.geomspace(f_min, f_max, num=32)
    scales = pywt.frequency2scale(wavelet, freqs_target / kwargs["fs"])
    W, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1 / kwargs["fs"], axis=-1)
    # W shape: (n_scales, ..., n_times) → transpose to (..., n_scales, n_times)
    W = np.moveaxis(W, 0, -2)
    P = np.mean(W**2, axis=-1)  # Collapse time → (..., n_scales)
    return freqs, P
```

### Pros
- Zero new files; `test_spectral.py` already covers the pattern
- `spectral_bands_power` and other spectral features work on wavelet output via `@feature_predecessor(spectral_preprocessor, wavelet_preprocessor)`
- Familiar pattern for consumers

### Cons
- **Conceptually wrong**: wavelets are not a spectral method; Welch and CWT are different algorithms for different purposes
- Permanently loses time-frequency locality (the main advantage of wavelets)
- `spectral.py` grows in scope and responsibility
- A "fake" frequency axis (log-spaced scales ≠ FFT bins) masquerades as equivalent to PSD
- Future time-frequency features (e.g., `wavelet_instantaneous_power(t, f, W)`) cannot be added without breaking the shape contract

---

## Option 2: Dedicated `wavelet.py` module (Recommended)

**Idea**: New `eegdash/features/feature_bank/wavelet.py` module with its own preprocessor chain and tests. Follows the exact same structure as `complexity.py` or `spectral.py`.

### Computation graph

```
Raw Signal (x)
    └─→ wavelet_preprocessor (freqs, W)           freqs: (n_scales,), W: (..., n_scales, n_times)
            ├─→ wavelet_band_energy_preprocessor  (freqs, E)  (..., n_scales)   [collapses time]
            │       ├─→ wavelet_bands_power        @univariate_feature → dict of band sums
            │       └─→ wavelet_relative_power     @univariate_feature → dict of normalized bands
            │
            ├─→ wavelet_entropy                   @univariate_feature  (Shannon on scale distribution)
            └─→ wavelet_variance                  @univariate_feature  (std across scales)
```

### File structure

```
eegdash/features/feature_bank/
    wavelet.py              ← NEW
    spectral.py             ← untouched
    utils.py                ← add wavelet_kwargs decorator + scale utilities

tests/unit_tests/features/feature_bank/
    test_wavelet.py         ← NEW
    test_spectral.py        ← untouched
```

### Preprocessor design

**`wavelet_preprocessor`** (root, depends on raw signal):
- Uses `pywt.cwt` with Morlet wavelet (default, most spectral-like)
- Returns `(freqs, W)`:
  - `freqs`: `(n_scales,)` — center frequencies in Hz
  - `W`: `(..., n_scales, n_times)` — complex or real CWT coefficients
- Parametrized by `wavelet`, `f_min`, `f_max`, `n_scales`
- Needs a `@wavelet_kwargs` decorator (mirrors `@spectral_kwargs`) to inject `fs`, defaults

**`wavelet_band_energy_preprocessor`** (depends on `wavelet_preprocessor`):
- Collapses time axis: `E = (|W|² ).mean(axis=-1)` → `(..., n_scales)`
- Returns `(freqs, E)` — now shape-compatible with spectral `(f, p)` 
- This is the bridge: features needing only scale-level statistics use this

### Feature functions

```python
@feature_predecessor(wavelet_band_energy_preprocessor)
@univariate_feature
def wavelet_bands_power(freqs, E, /, bands=DEFAULT_FREQ_BANDS):
    return utils.reduce_freq_bands(freqs, E, bands, np.sum)  # reuses existing utility


@feature_predecessor(wavelet_band_energy_preprocessor)
@univariate_feature
def wavelet_relative_power(freqs, E, /, bands=DEFAULT_FREQ_BANDS):
    total = E.sum(axis=-1, keepdims=True)
    return utils.reduce_freq_bands(freqs, E / total, bands, np.sum)


@feature_predecessor(wavelet_band_energy_preprocessor)
@univariate_feature
def wavelet_entropy(freqs, E, /):
    p = E / E.sum(axis=-1, keepdims=True)
    idx = p > 0
    plogp = np.zeros_like(p)
    plogp[idx] = p[idx] * np.log(p[idx])
    return -np.sum(plogp, axis=-1)


@feature_predecessor(wavelet_preprocessor)
@univariate_feature
def wavelet_variance(freqs, W, /):
    return np.std(np.abs(W), axis=(-2, -1))  # across scale and time
```

### Pros
- Clean separation of concerns — wavelet DAG is self-contained
- Preserves the full time-frequency representation `W` at the preprocessor level; time-varying features (e.g., instantaneous power per scale per epoch) can be added later without any breaking change
- Mirrors the exact module structure of `complexity.py` — easy to onboard
- `utils.reduce_freq_bands` is reused — wavelet band power is defined identically to spectral band power
- Test structure mirrors `test_complexity.py` pattern
- The `feature_bank/__init__.py` registration is a 3-line addition

### Cons
- One new source file, one new test file (minor overhead)
- Cannot directly share a feature function between spectral and wavelet without a multi-predecessor declaration (acceptable)

---

## Recommendation

**Option 2** (new `wavelet.py` module). The two-level preprocessor chain (`wavelet_preprocessor` → `wavelet_band_energy_preprocessor`) is the right design because:
1. It preserves the full CWT at the top level for future time-varying features
2. It presents a spectral-compatible `(freqs, E)` shape at the second level, enabling reuse of `utils.reduce_freq_bands`
3. It follows every established pattern in the codebase

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `eegdash/features/feature_bank/wavelet.py` | CREATE — full module |
| `tests/unit_tests/features/feature_bank/test_wavelet.py` | CREATE — test suite |
| `eegdash/features/feature_bank/__init__.py` | MODIFY — add exports |
| `eegdash/features/feature_bank/utils.py` | MODIFY — add `wavelet_kwargs` decorator |

---

## Existing Utilities to Reuse

| Utility | Location | Reuse point |
|---------|----------|-------------|
| `utils.reduce_freq_bands` | `feature_bank/utils.py` | `wavelet_bands_power`, `wavelet_relative_power` |
| `utils.DEFAULT_FREQ_BANDS` | `feature_bank/utils.py` | default `bands` parameter |
| `utils.slice_freq_band` | `feature_bank/utils.py` | trim CWT output to f_min/f_max |
| `@feature_predecessor` | `decorators.py` | all functions |
| `@univariate_feature` | `decorators.py` | all scalar features |

---

## Verification

```bash
# Run new wavelet tests
pytest tests/unit_tests/features/feature_bank/test_wavelet.py -v

# Confirm spectral tests still pass (no regression)
pytest tests/unit_tests/features/feature_bank/test_spectral.py -v

# Full feature bank
pytest tests/unit_tests/features/ -v
```

Test strategy:
- Synthetic sine wave at known frequency → verify `wavelet_bands_power` puts power in the correct band
- Behavioral: white noise should yield higher `wavelet_entropy` than a single-frequency signal
- Shape assertions: output shape matches `(n_channels,)` for `@univariate_feature` functions
- Edge cases: single channel, batched input
