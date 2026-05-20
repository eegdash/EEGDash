# Plan: Wavelet Feature Extraction for EEGDash

## Context

The `eegdash` feature bank uses a DAG-based computation graph via `@feature_predecessor` decorators. Each domain (spectral, complexity, signal) lives in its own module. We are adding wavelet features as a new dedicated `wavelet.py` module — the same pattern as `complexity.py` and `spectral.py`.

---

## Preprocessor Design

```python
def wavelet_preprocessor(
    x,
    frequency_bands,  # list of (f_low, f_high) Hz tuples — consistent with rest of EEGDash
    wavelet="morl",  # pywt wavelet name, defaults to Morlet
    *,
    _metadata,  # carries sfreq for scale conversion
): ...
```

**Transform:** CWT via `pywt.cwt`

**Output shape:** `(n_bands, n_channels, n_times)` — time dimension preserved, one slice per frequency band.

**`wavelet` parameter options:**

| Wavelet | pywt name | Use case |
|---|---|---|
| Morlet (default) | `'morl'` | General EEG, enables phase features |
| Complex Morlet | `'cmor{B}-{C}'` | Explicit bandwidth/centre control |
| Mexican hat | `'mexh'` | Spike and transient detection |
| Daubechies db4 | `'db4'` | Fast energy features only |
| Symlet 4 | `'sym4'` | Like db4, more symmetric |

Real wavelets (db4, mexh) return real coefficients — energy features work, phase features raise a clear error. Complex wavelets (morl, cmor) return complex coefficients — all features available including PLV, PAC, and wavelet coherence.

[Gal's note]: we should think what is the best way to handle this (not all wavelets are capable of producing the continued steps).

### Why CWT, not DWT

DWT returns a list of arrays of different lengths (N/2, N/4, N/8 …) — incompatible with EEGDash's preprocessor shape contract. It also requires users to specify levels rather than frequencies in Hz, making it dataset-specific and error-prone across different sampling rates.

---

## Computation Graph

```
Raw Signal (x)
    └─→ wavelet_preprocessor(x, frequency_bands, wavelet, *, _metadata)
            │   Returns: W  shape (n_bands, n_channels, n_times)
            │
            ├─→ wavelet_entropy          @univariate_feature  −Σ p_j log p_j over band energies
            ├─→ wavelet_relative_power   @univariate_feature  per-band energy / total energy
            ├─→ wavelet_skewness         @univariate_feature  skewness of |W| per band
            ├─→ wavelet_kurtosis         @univariate_feature  kurtosis of |W| per band
            ├─→ wavelet_holder_exponent  @univariate_feature  slope of log|W| vs log(scale)
            ├─→ wavelet_plv              @bivariate_feature   phase locking between channel pairs
            ├─→ wavelet_pac              @univariate_feature  phase-amplitude coupling per band-pair
            └─→ wavelet_coherence        @bivariate_feature   time-averaged cross-channel coherence
```

---

## Priority Feature Implementations

In order of impact:

1. **Wavelet Entropy (Rosso)** — `−Σ p_j log p_j` over relative band energies. Most cited wavelet feature in clinical EEG.
2. **Instantaneous phase per band** — `angle(W(s,t))` from complex Morlet output. Feeds PLV and PAC.
3. **Wavelet PLV** — phase locking between channel pairs, more accurate than filter + Hilbert.
4. **Wavelet PAC** — phase-amplitude coupling between band pairs, using Modulation Index (Tort et al.).
5. **Wavelet coherence** — cross-channel time-frequency coherence, bivariate, time-averaged per band.
6. **Relative energy ratios** — per-band energy / total energy, scale-invariant band power.
7. **Skewness and kurtosis of coefficients** — pathology-sensitive distributional features.
8. **Hölder regularity exponent** — slope of log|W(s,t)| vs log(s), wavelet-native fractal measure.

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `eegdash/features/feature_bank/wavelet.py` | CREATE — full module |
| `tests/unit_tests/features/feature_bank/test_wavelet.py` | CREATE — test suite |
| `eegdash/features/feature_bank/__init__.py` | MODIFY — add exports |

---

## Existing Utilities to Reuse

| Utility | Location | Reuse point |
|---------|----------|-------------|
| `utils.DEFAULT_FREQ_BANDS` | `feature_bank/utils.py` | default `frequency_bands` parameter |
| `@feature_predecessor` | `decorators.py` | all functions |
| `@univariate_feature` | `decorators.py` | all scalar features |
| `@bivariate_feature` + `@channel_pairer_undirected` | `decorators.py` | PLV, coherence |

---

## Verification

```bash
pytest tests/unit_tests/features/feature_bank/test_wavelet.py -v
pytest tests/unit_tests/features/feature_bank/test_spectral.py -v  # regression check
pytest tests/unit_tests/features/ -v
```

Test strategy:
- Synthetic sine wave at known frequency → verify band energy is concentrated in the matching band
- White noise should yield higher `wavelet_entropy` than a single-frequency signal
- Complex wavelet (morl): `wavelet_plv` returns 1.0 for identical channels, ~0 for independent noise
- Real wavelet (mexh): `wavelet_plv` raises a clear error (no phase information)
- Shape assertions: output is `(n_channels,)` for `@univariate_feature`, `(n_channel_pairs,)` for `@bivariate_feature`
- Edge cases: single channel, batched input
