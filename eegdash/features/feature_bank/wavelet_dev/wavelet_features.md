# Wavelet Features for EEG — Reference Guide

This document catalogues all known wavelet-domain features relevant to EEG analysis, organized by family. Each entry includes an explanation, the defining formula, an importance rating for EEG decoding tasks (1 = rarely useful, 5 = high impact), and implementation notes where relevant.

Transform abbreviations used throughout: **DWT** = Discrete Wavelet Transform, **CWT** = Continuous Wavelet Transform, **WPD** = Wavelet Packet Decomposition. Coefficient notation: `w(s, t)` denotes the wavelet coefficient at scale `s` and time `t`; `w_j[n]` denotes the DWT detail coefficient at level `j` and sample `n`.

---

## 1. Energy and Power Features

### 1.1 Coefficient Energy

**Explanation**
The total energy carried by wavelet coefficients at a given scale or decomposition level. Directly analogous to band power in the spectral domain, but computed from wavelet coefficients rather than a Fourier-based PSD. For DWT, computed separately for each detail level and the final approximation level. For CWT, computed for each scale (or frequency band). This is the single most widely used wavelet feature in EEG literature and forms the foundation for all ratio and entropy features below.

**Formula**

$$E_j = \sum_{n} |w_j[n]|^2$$

For CWT at scale $s$:

$$E(s) = \int |W(s, t)|^2 \, dt \approx \sum_{t} |w(s, t)|^2$$

**Importance:** ★★★★★

**Notes**
- Equivalent to variance of zero-mean coefficients.
- For DWT with db4, level 1 corresponds to the highest frequency band; deeper levels correspond to progressively lower frequencies.
- In CWT, scale and frequency are inversely related: $f \approx f_c / (s \cdot \Delta t)$ where $f_c$ is the central frequency of the mother wavelet.
- Already implemented in `braindecode-features` (as `power`) and in `brainfeatures` via mne-features (`wavelet_coef_energy` with db4).

---

### 1.2 Relative Energy (Energy Ratio)

**Explanation**
The fraction of total signal energy carried by each scale or level. Normalises coefficient energy by the sum across all scales, making the feature invariant to the absolute amplitude of the signal. This is critical for cross-subject comparisons where electrode impedance, skull thickness, or amplifier gain can create large inter-subject amplitude differences. A subject-level normalisation is implicit in the ratio, so the feature characterises the *shape* of the energy distribution across scales rather than its magnitude.

**Formula**

$$p_j = \frac{E_j}{\sum_{k} E_k}$$

where the sum runs over all decomposition levels (detail levels plus the approximation).

**Importance:** ★★★★☆

**Notes**
- By construction, $\sum_j p_j = 1$, so the full set of relative energies lies on a simplex. If you include all levels as features, they are linearly dependent — drop one or be aware of this in regularised models.
- Present in `brainfeatures` only in the spectral (Fourier) domain as `power_ratio`. Not implemented in the wavelet domain in any of the three codebases.
- Closely related to the input needed for Wavelet Entropy (Rosso) — see section 3.2.

---

### 1.3 Teager-Kaiser Energy

**Explanation**
The Teager-Kaiser operator is a nonlinear energy measure that is sensitive to both instantaneous amplitude *and* instantaneous frequency of an oscillation simultaneously. Applied to wavelet detail coefficients, it gives a per-band measure of modulated energy that responds faster to transients than squared-amplitude energy. Particularly useful for detecting rapid-onset events such as epileptic spikes or sleep K-complexes, where the frequency of an oscillation changes abruptly.

**Formula**

For a discrete signal $x[n]$, the Teager-Kaiser operator is:

$$\Psi\{x[n]\} = x[n]^2 - x[n-1] \cdot x[n+1]$$

Applied to the detail coefficients $w_j[n]$ at level $j$:

$$TKE_j = \sum_{n} \Psi\{w_j[n]\} = \sum_{n} \left( w_j[n]^2 - w_j[n-1] \cdot w_j[n+1] \right)$$

**Importance:** ★★★☆☆

**Notes**
- The standard squared-energy operator measures amplitude only. TKE mixes amplitude and frequency, making it a richer signal descriptor for transient EEG phenomena.
- Available in `brainfeatures` via mne-features as `teager_kaiser_energy`. Not in `braindecode-features` or EEGDash.
- Requires at least 3 samples per coefficient sequence; edge samples are typically excluded.
- Can be computed on CWT coefficients at each scale as well, giving a time-varying TKE scalogram.

---

## 2. Statistical Moment Features

### 2.1 Variance of Coefficients

**Explanation**
The second central moment of the coefficient distribution at each level. For zero-mean detail coefficients, variance equals energy divided by the number of coefficients. While numerically close to coefficient energy, variance has slightly different normalisation and is the preferred form when comparing across windows of different lengths.

**Formula**

$$\sigma_j^2 = \frac{1}{N_j} \sum_{n} \left(w_j[n] - \mu_j\right)^2$$

**Importance:** ★★★★☆

**Notes**
- Already in both `braindecode-features` and `brainfeatures`.
- The variance of approximation coefficients at the deepest DWT level is equivalent to the low-frequency signal power — it behaves like a delta-band feature.

---

### 2.2 Skewness of Coefficients

**Explanation**
The third standardised central moment of the coefficient distribution. Measures asymmetry: positive skewness means a long tail of large positive coefficients (sharp positive peaks in the band), negative skewness means large negative deviations. In EEG, pathological events like epileptic spikes consistently produce asymmetric coefficient distributions at the scale corresponding to their dominant frequency. Resting healthy EEG tends to have near-zero skewness at most scales.

**Formula**

$$\gamma_j = \frac{1}{N_j \sigma_j^3} \sum_{n} \left(w_j[n] - \mu_j\right)^3$$

**Importance:** ★★★☆☆

**Notes**
- Not present in the wavelet module of any of the three codebases. Present in the time-domain signal module of EEGDash (`signal_skewness`).
- Implementation is straightforward: `scipy.stats.skew(coefficients, axis=-1)` applied to the coefficient array at each level.
- Particularly discriminative for: epilepsy vs normal, sleep stage N2 (K-complexes) vs other stages, artefact detection.

---

### 2.3 Kurtosis of Coefficients

**Explanation**
The fourth standardised central moment, measuring the "tailedness" of the coefficient distribution. High kurtosis (leptokurtic) means many near-zero coefficients punctuated by occasional very large ones — the signature of a sparse, transient oscillatory event. Low kurtosis (platykurtic) indicates a more uniform distribution, consistent with sustained broadband noise. Wavelet coefficient distributions in EEG are typically super-Gaussian (heavy-tailed) due to the sparse, burst-like nature of neural oscillations; kurtosis quantifies how far this departs from Gaussianity.

**Formula**

$$\kappa_j = \frac{1}{N_j \sigma_j^4} \sum_{n} \left(w_j[n] - \mu_j\right)^4 - 3$$

(Excess kurtosis; zero for a Gaussian distribution.)

**Importance:** ★★★★☆

**Notes**
- Not present in the wavelet module of any of the three codebases.
- High-kurtosis detail coefficients at beta/gamma scales are associated with muscle artefact (EMG), which has a very impulsive wavelet representation. This makes kurtosis useful for artefact rejection as well as classification.
- In seizure detection, kurtosis at high-frequency levels spikes dramatically at seizure onset due to rapid, high-amplitude oscillations.

---

### 2.4 Interquartile Range of Coefficients

**Explanation**
The difference between the 75th and 25th percentile of the coefficient distribution. A robust measure of spread that is insensitive to the extreme outlier coefficients that arise from artefacts or transient events. Complements variance as a robust alternative that does not square large values.

**Formula**

$$IQR_j = Q_{75}\left(\{w_j[n]\}\right) - Q_{25}\left(\{w_j[n]\}\right)$$

**Importance:** ★★★☆☆

**Notes**
- Not implemented in the wavelet module of any of the three codebases.
- Especially useful for EEG datasets without aggressive artefact rejection, where extreme coefficients would distort variance estimates.
- Implementation: `np.percentile(coefficients, 75, axis=-1) - np.percentile(coefficients, 25, axis=-1)`.

---

## 3. Entropy Features

### 3.1 Shannon Entropy of Coefficients

**Explanation**
Treats the normalised squared coefficients as a probability distribution and computes the Shannon entropy. Measures how evenly the energy is spread across time (within a given scale) or across scales (within a given window). A high Shannon entropy at a particular scale means the energy is spread uniformly over the analysis window — consistent with sustained, continuous oscillation. A low entropy means the energy is concentrated in a few time points — consistent with a transient burst.

**Formula**

Let $p_n = |w_j[n]|^2 / E_j$ be the normalised squared coefficient at position $n$ (treating the time axis as a discrete probability distribution):

$$H_{Sh,j} = -\sum_{n} p_n \log p_n$$

**Importance:** ★★★☆☆

**Notes**
- Distinct from Wavelet Entropy (Rosso) below: Shannon entropy here is computed *within* a scale (over time), whereas Rosso entropy is computed *across* scales.
- Not implemented in the wavelet module of any of the three codebases. `spectral_entropy` in EEGDash computes this in the Fourier domain.

---

### 3.2 Wavelet Entropy (Rosso)

**Explanation**
Proposed by Rosso et al. (2001) specifically for EEG analysis. Rather than treating the time axis as a probability distribution within one scale, it treats the *relative energy distribution across scales* as a probability distribution and computes its Shannon entropy. A flat distribution (equal energy at all scales = broad-spectrum noise) gives maximum wavelet entropy. A peaked distribution (all energy at one scale = narrow-band oscillation) gives low wavelet entropy. This makes it a single scalar summary of how "complex" or "disordered" the overall spectral structure of a segment is.

**Formula**

$$WE = -\sum_{j} p_j \log p_j$$

where $p_j = E_j / \sum_k E_k$ is the relative energy at level $j$ (see section 1.2).

**Importance:** ★★★★★

**Notes**
- Extensively validated in EEG: tracks anaesthesia depth, sleep stages, mental workload, and cognitive load. One of the most cited wavelet features in the clinical EEG literature.
- Not implemented in any of the three codebases.
- Requires DWT (or WPD) for the "across scales" interpretation to be well-defined. Can be extended to CWT by discretising the scale axis into frequency bands.
- Reference: Rosso OA et al. (2001). *Wavelet entropy: a new tool for analysis of short duration brain electrical signals.* Journal of Neuroscience Methods, 105(1), 65–75.

---

### 3.3 Permutation Entropy of Coefficients

**Explanation**
Treats the time-ordered sequence of wavelet coefficients at a given scale as a time series and computes its permutation entropy. The coefficient sequence is embedded into ordinal patterns (the rank ordering of successive groups of $m$ coefficients), and the Shannon entropy of the resulting pattern distribution is computed. This captures the temporal complexity of the wavelet representation at each frequency band — whether the coefficients evolve in a predictable or chaotic fashion over time.

**Formula**

For embedding dimension $m$ and lag $\tau$, define the ordinal pattern $\pi$ of $\mathbf{w} = [w_j[n], w_j[n+\tau], \ldots, w_j[n+(m-1)\tau]]$ as the permutation that sorts its elements. Let $p(\pi)$ be the relative frequency of each of the $m!$ patterns:

$$H_{PE,j} = -\sum_{\pi} p(\pi) \log p(\pi)$$

**Importance:** ★★★☆☆

**Notes**
- Not implemented in the wavelet module of any of the three codebases. `braindecode-features` has permutation entropy in the time domain.
- Computationally cheap, robust to noise, and parameter-light (only $m$ and $\tau$ needed). Recommended $m = 3$ to $6$ for EEG signals.
- Combines two levels of analysis: the wavelet decomposes by frequency, and the permutation entropy characterises the *temporal dynamics* at each frequency band.

---

## 4. Regularity and Fractal Features

### 4.1 Hölder / Lipschitz Regularity Exponent

**Explanation**
The Hölder exponent $\alpha$ characterises how smooth a signal is at a point. In the CWT, the local regularity of a signal is encoded in how the coefficient modulus decays as the scale decreases. A smooth signal (large $\alpha$) has coefficients that decay rapidly towards fine scales; an irregular or fractal signal (small $\alpha$) has coefficients that decay slowly or not at all. This makes the Hölder exponent the wavelet-native equivalent of the fractal dimension measures already in EEGDash (`higuchi`, `katz`, `petrosian`, DFA), but computed more directly from the multiscale structure of the CWT rather than from box-counting or R/S analysis.

**Formula**

If the CWT coefficients satisfy the decay law:

$$|W(s, t_0)| \leq C \cdot s^{\alpha + 1/2} \quad \text{as } s \to 0$$

then $\alpha$ is the Hölder exponent at $t_0$. In practice, $\alpha$ is estimated as the slope of $\log |W(s, t))|$ versus $\log s$ across a range of scales, for each time point $t$:

$$\hat{\alpha}(t) = \text{slope of } \log|W(s,t)| \text{ vs } \log s$$

The global regularity exponent for a window is typically the mean or median of $\hat{\alpha}(t)$ over $t$.

**Importance:** ★★★★☆

**Notes**
- Not implemented in any of the three codebases.
- Directly related to the Hurst exponent ($H = \alpha + 1/2$ for self-similar processes) and to the DFA scaling exponent, but derived from the wavelet multiscale structure rather than statistical fluctuations.
- Requires a complex analytic wavelet (Morlet, Morse) or the modulus of the CWT with a real wavelet. The estimate is only reliable within the cone of influence (away from signal boundaries).
- Useful for: characterising the 1/f nature of EEG, detecting transitions in brain state (the exponent increases during deep sleep and anesthesia as signals become smoother).
- Reference: Mallat S & Hwang WL (1992). *Singularity detection and processing with wavelets.* IEEE Transactions on Information Theory, 38(2), 617–643.

---

### 4.2 Wavelet Fractal Dimension

**Explanation**
The fractal dimension of a signal can be estimated from the scaling behaviour of its wavelet coefficient variance across levels. If the variance of detail coefficients scales as a power law with the scale $s$, the exponent determines the fractal dimension. This is more robust than box-counting methods for non-stationary EEG because the wavelet transform handles trend non-stationarity explicitly (through the vanishing moments of the wavelet).

**Formula**

For DWT detail levels $j = 1, \ldots, J$, compute the log-variance of coefficients at each level:

$$\log \sigma_j^2 \approx \beta \cdot j + \text{const}$$

The fractal dimension $D$ is then derived from the slope $\beta$ of this log-linear relationship:

$$D = \frac{5 - \beta}{2} \quad \text{(for a 1D signal)}$$

**Importance:** ★★★☆☆

**Notes**
- Not implemented in any of the three codebases.
- Complementary to but distinct from Higuchi and Katz fractal dimensions: those are computed in the time domain, this is computed in the wavelet domain and has a direct spectral interpretation.
- The slope $\beta$ is related to the spectral exponent $\gamma$ of a 1/f process by $\beta = \gamma - 1$ (for a DWT with sufficient vanishing moments).
- For EEG, the wavelet fractal dimension is particularly stable across epochs and robust to artefacts because the DWT filters isolate frequency bands.

---

## 5. Morphology Features

### 5.1 Peak-to-Peak Amplitude of Coefficients

**Explanation**
The range of coefficient values at a given scale: the maximum minus the minimum. Captures the maximal excursion of wavelet-domain oscillatory activity within the analysis window. More sensitive to single large transient events than variance (which distributes the contribution of a spike across many terms). In seizure detection, a spike in peak-to-peak at a particular DWT level is often the earliest detectable change.

**Formula**

$$PtP_j = \max_n(w_j[n]) - \min_n(w_j[n])$$

**Importance:** ★★★☆☆

**Notes**
- Already implemented in `braindecode-features` as `value_range`.
- Simple to compute; acts as a soft outlier detector for the coefficient sequence at each scale.

---

### 5.2 Zero Crossings of Coefficients

**Explanation**
The number of times the wavelet detail coefficient sequence at a given level crosses zero within the analysis window. The expected zero-crossing rate for a bandpass signal is proportional to its centre frequency. Deviation from the expected rate — either excess crossings (noisy, high-frequency contamination at a low-frequency scale) or too few (sustained unidirectional deviation) — is diagnostically informative.

**Formula**

$$ZC_j = \sum_{n} \mathbf{1}\left[w_j[n] \cdot w_j[n+1] < 0\right]$$

**Importance:** ★★★☆☆

**Notes**
- Not in the wavelet module of any codebase, though `signal_zero_crossings` exists in EEGDash for raw signals.
- At DWT level $j$, the expected zero-crossing rate for a white noise input is $\approx N_j / 3$ crossings. Departures from this baseline are meaningful.
- Can detect: alpha burst cessation (sudden drop in zero crossings at alpha-scale coefficients), muscle artefact (excess zero crossings at high-frequency levels).

---

## 6. Time-Frequency Features (CWT Scalogram)

### 6.1 Instantaneous Amplitude Envelope per Band

**Explanation**
From the CWT using a complex (analytic) wavelet, the modulus $|W(s, t)|$ gives the instantaneous amplitude envelope of the signal at frequency $f \propto 1/s$ as a function of time. Rather than collapsing the time axis (as coefficient energy does), this preserves the temporal structure of amplitude fluctuations. Summary statistics of this amplitude envelope — its mean, variance, skewness — are then computed per frequency band. This is the wavelet equivalent of the filter + Hilbert amplitude envelope, but without the ringing artefacts introduced by sharp-cutoff bandpass filters.

**Formula**

$$A(f, t) = |W(s(f), t)|$$

where $s(f) = f_c / (f \cdot \Delta t)$ and $f_c$ is the central frequency of the mother wavelet. Per-band summary:

$$\bar{A}(f) = \frac{1}{T} \int A(f, t) \, dt$$

**Importance:** ★★★★☆

**Notes**
- Requires a complex mother wavelet: Morlet, Morse, or Paul. Cannot be computed from a real wavelet without losing phase information.
- The mean amplitude over time is equivalent to the square root of coefficient energy (up to normalisation). The *variance* of $A(f, t)$ over time is a new feature: it measures how much the band amplitude fluctuates, which is related to alpha-band rhythmicity in resting EEG.
- Not in any of the three codebases.

---

### 6.2 Instantaneous Phase per Band

**Explanation**
From a complex CWT, the phase $\angle W(s, t) = \arctan\left(\text{Im}(W) / \text{Re}(W)\right)$ gives the instantaneous phase of the signal at each scale and time point. This is more accurate than bandpass filtering followed by the Hilbert transform, because the wavelet acts as a Gaussian-windowed bandpass filter with smooth spectral edges, avoiding the frequency leakage and ringing artifacts that sharp FIR/IIR filters introduce. Instantaneous phase is the input to phase locking value, phase-amplitude coupling, and inter-trial phase coherence calculations.

**Formula**

$$\phi(f, t) = \angle W(s(f), t) = \arctan\left(\frac{\text{Im}(W(s(f), t))}{\text{Re}(W(s(f), t))}\right)$$

**Importance:** ★★★★★

**Notes**
- This is a **preprocessor output**, not a feature by itself. It is the input to PLV (section 8.2) and wavelet-based PAC (section 8.3).
- Not implemented in any of the three codebases.
- The key advantage over Hilbert-based phase: the Morlet wavelet in CWT is already a complex exponential modulated by a Gaussian window, so its output is analytic by construction. The Hilbert transform applied to a narrow bandpass signal approximates this, but suffers from edge effects and spectral leakage at the filter boundaries.

---

### 6.3 Scalogram Entropy

**Explanation**
Entropy computed over the full 2D time-frequency energy distribution from the CWT scalogram. Treats the normalised $|W(s, t)|^2$ as a 2D probability distribution over both time and scale, and computes its Shannon entropy. A sleeping brain with persistent, spatially concentrated alpha rhythm gives a low-entropy scalogram (energy concentrated at one band, sustained in time). A resting waking brain gives moderate entropy. A seizure gives very low entropy at onset (sudden concentration of energy at one scale) followed by high entropy (spread across scales).

**Formula**

Let $P(s, t) = |W(s, t)|^2 / \sum_{s',t'} |W(s', t')|^2$:

$$H_{scalogram} = -\sum_{s,t} P(s, t) \log P(s, t)$$

**Importance:** ★★★☆☆

**Notes**
- Not implemented in any of the three codebases.
- One scalar per channel per window. Provides a holistic summary of time-frequency structure without committing to fixed band boundaries.
- Computationally more expensive than per-scale features because the full scalogram must be retained.
- Can be decomposed along the time axis (entropy per time point, averaging over scale) or along the scale axis (entropy per scale, averaging over time) to give temporal or spectral complexity profiles.

---

### 6.4 Cone of Influence Masking

**Explanation**
In CWT, the wavelet at large scales (low frequencies) has a wide time support, which means that coefficients near the edges of the analysis window are influenced by data outside the window (zero-padding or reflection artefacts). The cone of influence (COI) defines, for each scale $s$, the time region within which the coefficients are reliable. Features computed on coefficients inside the COI are more trustworthy.

**Formula**

For a Morlet wavelet with parameter $\omega_0$, the e-folding time at scale $s$ is:

$$\tau_{COI}(s) = \sqrt{2} \cdot s$$

Coefficients at time $t$ and scale $s$ are inside the COI if $t \in [\tau_{COI}(s), T - \tau_{COI}(s)]$.

**Importance:** ★★★☆☆

**Notes**
- This is a **data quality filter**, not a feature. It should be applied before computing any CWT-based feature, particularly for low-frequency bands where the wavelet support is wide relative to the window length.
- None of the three codebases implement COI masking. In practice, windows that are too short for reliable low-frequency CWT analysis (e.g. 1-second windows for delta-band CWT with a Morlet wavelet) will silently return artefactual features.
- Rule of thumb: for reliable CWT at frequency $f$, the window must be at least $4/f$ seconds (four cycles of the slowest frequency).

---

## 7. Cross-Scale Features

### 7.1 Inter-Scale Correlation

**Explanation**
The Pearson correlation between the coefficient sequences at two adjacent DWT levels. Strong correlation between level $j$ and level $j+1$ indicates a broad-band event that spans both scales simultaneously — for example, a K-complex in sleep EEG, which appears as energy spread across delta and theta levels, or a gamma burst, which spans multiple high-frequency levels. Near-zero inter-scale correlation indicates activity that is narrowband and scale-specific.

**Formula**

$$\rho_{j, j+1} = \frac{\text{Cov}(w_j, w_{j+1}^{up})}{\sigma_j \cdot \sigma_{j+1}^{up}}$$

where $w_{j+1}^{up}$ denotes the level $j+1$ coefficients upsampled to match the length of level $j$ (by linear interpolation or coefficient repetition).

**Importance:** ★★★☆☆

**Notes**
- Not implemented in any of the three codebases.
- Can be generalised to non-adjacent levels to form an inter-scale correlation matrix.
- Requires careful handling of the different lengths of coefficient sequences at each DWT level (each level has half the samples of the level above).

---

### 7.2 WPD Best-Basis Energy Distribution

**Explanation**
In Wavelet Packet Decomposition, the signal is represented in a full binary tree of subbands. The best-basis algorithm selects the set of nodes (subbands) that minimises a chosen cost function (typically Shannon or log energy entropy), giving the most efficient representation of the signal. The energy distribution across the selected best-basis nodes is a compact descriptor of the signal's spectral structure, with adaptive frequency resolution — unlike DWT which fixes the frequency bands, WPD finds the bands most relevant for the signal at hand.

**Formula**

The best-basis is the set $\mathcal{B}^*$ minimising:

$$\mathcal{B}^* = \arg\min_{\mathcal{B}} \sum_{(j,k) \in \mathcal{B}} H(w_{j,k})$$

where $H$ is the chosen cost function (entropy) and $(j,k)$ indexes level $j$, node $k$ in the WPD tree. The feature vector is then $\{E_{j,k} : (j,k) \in \mathcal{B}^*\}$.

**Importance:** ★★★☆☆

**Notes**
- Not implemented in any of the three codebases.
- Requires `pywt.WaveletPacket` (available in PyWavelets). The best-basis node selection can be done with `pywt.WaveletPacket`'s tree decomposition.
- The resulting feature vector has variable length across subjects or sessions (different best bases may be selected). This complicates downstream sklearn pipelines that expect fixed-length inputs. A workaround is to fix the tree depth and use all nodes at that depth.
- Reference: Coifman RR & Wickerhauser MV (1992). *Entropy-based algorithms for best basis selection.* IEEE Transactions on Information Theory, 38(2), 713–718.

---

## 8. Cross-Channel Wavelet Features

### 8.1 Wavelet Coherence

**Explanation**
The wavelet analog of spectral coherence. Measures the degree of linear correlation between two channels in the time-frequency plane, giving a coherence value at each scale and time point. Unlike Fourier coherence (which averages over the entire analysis window), wavelet coherence is local in time and can detect transient synchronisation episodes between channels. The time-averaged wavelet coherence at each scale is the wavelet equivalent of EEGDash's `connectivity_magnitude_square_coherence`, but more robust to non-stationary EEG.

**Formula**

The cross-wavelet spectrum of signals $x$ and $y$ is:

$$W_{xy}(s, t) = W_x(s, t) \cdot \overline{W_y(s, t)}$$

The wavelet coherence is:

$$C_{xy}(s, t) = \frac{\left|\mathcal{S}\left(W_{xy}(s, t)\right)\right|^2}{\mathcal{S}\left(|W_x(s, t)|^2\right) \cdot \mathcal{S}\left(|W_y(s, t)|^2\right)}$$

where $\mathcal{S}$ denotes a smoothing operator applied in both scale and time. The feature is $\bar{C}_{xy}(s) = \frac{1}{T} \int C_{xy}(s, t) \, dt$ per frequency band.

**Importance:** ★★★★★

**Notes**
- Not implemented in any of the three codebases.
- Requires a complex mother wavelet (Morlet or Morse) so that the cross-wavelet spectrum is well-defined.
- The smoothing operator $\mathcal{S}$ is essential: without it, $C_{xy}(s,t)$ is always 1 (trivially). Typically smoothed with a Gaussian in time and a boxcar in scale.
- Bivariate feature: produces one value per channel pair per frequency band (same structure as `connectivity_magnitude_square_coherence` in EEGDash). Slots naturally into EEGDash's `@bivariate_feature` + `@channel_pairer_undirected` decorator pattern.
- Reference: Grinsted A, Moore JC & Jevrejeva S (2004). *Application of the cross wavelet transform and wavelet coherence to geophysical time series.* Nonlinear Processes in Geophysics, 11(5/6), 561–566.

---

### 8.2 Wavelet Phase Locking Value (wPLV)

**Explanation**
The phase locking value computed from instantaneous phases extracted via CWT rather than the Hilbert transform. For each frequency band, the instantaneous phase of two channels is extracted from the complex CWT at the corresponding scale. The phase difference is then averaged over time in complex exponential form. This gives a measure of phase synchrony between channels that is more accurate than filter + Hilbert PLV, particularly at low frequencies where bandpass filter ringing is most problematic.

**Formula**

Let $\phi_x(f, t)$ and $\phi_y(f, t)$ be the instantaneous phases from the complex CWT of channels $x$ and $y$ at frequency $f$:

$$wPLV(f) = \left| \frac{1}{T} \sum_{t} e^{i(\phi_x(f,t) - \phi_y(f,t))} \right|$$

**Importance:** ★★★★☆

**Notes**
- Not implemented in any of the three codebases as a wavelet-domain feature. `brainfeatures` has a Hilbert-based PLV.
- Direct upgrade path: replace the `signal_filter_preprocessor` + `signal_hilbert_preprocessor` chain used for PLV with a `wavelet_complex_preprocessor` and compute the angle directly.
- Bivariate feature (one value per channel pair per band). Fits EEGDash's `@bivariate_feature` decorator.
- Values in [0, 1]: 0 = no phase locking, 1 = perfect phase locking.

---

### 8.3 Wavelet-Based Phase-Amplitude Coupling (wPAC)

**Explanation**
Phase-Amplitude Coupling measures whether the amplitude of high-frequency oscillations is modulated by the phase of low-frequency oscillations. The wavelet-based version replaces the filter + Hilbert pipeline used in `braindecode-features` with CWT-derived instantaneous phase and amplitude. This avoids the two main failure modes of the Hilbert approach: (1) the ringing of the bandpass filter contaminates the phase signal, and (2) the Hilbert transform's implicit assumption of a narrow-band signal is violated at EEG band boundaries. With a complex CWT using a Morlet wavelet, both phase and amplitude are extracted from the analytic signal directly at the target frequency.

**Formula**

Extract from the CWT: phase of the low-frequency band $f_{low}$ and amplitude of the high-frequency band $f_{high}$:

$$\phi_{low}(t) = \angle W(s(f_{low}), t), \qquad A_{high}(t) = |W(s(f_{high}), t)|$$

Bin $\phi_{low}(t)$ into $K$ phase bins $[\phi_k, \phi_{k+1})$. Compute mean amplitude per bin:

$$\bar{A}_{high}(\phi_k) = \frac{1}{|\mathcal{T}_k|} \sum_{t \in \mathcal{T}_k} A_{high}(t)$$

where $\mathcal{T}_k = \{t : \phi_{low}(t) \in [\phi_k, \phi_{k+1})\}$.

The coupling statistic is the peak-to-peak range of bin-mean amplitudes (as in `braindecode-features`):

$$wPAC(f_{low}, f_{high}) = \max_k \bar{A}_{high}(\phi_k) - \min_k \bar{A}_{high}(\phi_k)$$

A more principled alternative is the Modulation Index (Tort et al., 2010):

$$MI = \frac{H_{max} - H\left(\{\bar{A}_{high}(\phi_k) / \sum_k \bar{A}_{high}(\phi_k)\}\right)}{H_{max}}$$

where $H$ is the Shannon entropy and $H_{max} = \log K$.

**Importance:** ★★★★★

**Notes**
- Not implemented in any of the three codebases in wavelet form. `braindecode-features` has the Hilbert-based version.
- This is the highest-priority wavelet connectivity feature for EEG: PAC is implicated in memory consolidation (theta-gamma), motor control (beta-gamma), and is disrupted in Parkinson's disease, epilepsy, and working memory disorders.
- Bivariate in the band dimension (one value per band-pair per channel). Fits the EEGDash decorator system as `@univariate_feature` (per channel) with the band-pair loop handled externally, mirroring the `braindecode-features` architecture.
- The Modulation Index formulation is preferred over peak-to-peak because it is bounded in [0, 1] and has a statistical interpretation (deviation from uniform distribution).
- Reference: Tort et al. (2010). *Measuring phase-amplitude coupling between neuronal oscillations of different frequencies.* Journal of Neurophysiology, 104(2), 1195–1210.

---

## Summary Table

| Feature | Family | Transform | Importance | In any codebase? |
|---|---|---|---|---|
| Coefficient Energy | Energy | DWT / CWT / WPD | ★★★★★ | ✓ braindecode-features, brainfeatures |
| Relative Energy Ratio | Energy | DWT / CWT / WPD | ★★★★☆ | ✗ |
| Teager-Kaiser Energy | Energy | DWT / CWT | ★★★☆☆ | ✓ brainfeatures (mne) |
| Variance of Coefficients | Statistical | DWT / CWT / WPD | ★★★★☆ | ✓ braindecode-features, brainfeatures |
| Skewness of Coefficients | Statistical | DWT / CWT / WPD | ★★★☆☆ | ✗ |
| Kurtosis of Coefficients | Statistical | DWT / CWT / WPD | ★★★★☆ | ✗ |
| IQR of Coefficients | Statistical | DWT / CWT / WPD | ★★★☆☆ | ✗ |
| Shannon Entropy of Coefficients | Entropy | DWT / CWT / WPD | ★★★☆☆ | ✗ |
| Wavelet Entropy (Rosso) | Entropy | DWT / WPD | ★★★★★ | ✗ |
| Permutation Entropy of Coefficients | Entropy | DWT / CWT / WPD | ★★★☆☆ | ✗ |
| Hölder Regularity Exponent | Regularity | CWT | ★★★★☆ | ✗ |
| Wavelet Fractal Dimension | Regularity | DWT / CWT | ★★★☆☆ | ✗ |
| Peak-to-Peak of Coefficients | Morphology | DWT / CWT / WPD | ★★★☆☆ | ✓ braindecode-features |
| Zero Crossings of Coefficients | Morphology | DWT / CWT / WPD | ★★★☆☆ | ✗ |
| Instantaneous Amplitude Envelope | Time-Frequency | CWT | ★★★★☆ | ✗ |
| Instantaneous Phase | Time-Frequency | CWT | ★★★★★ | ✗ |
| Scalogram Entropy | Time-Frequency | CWT | ★★★☆☆ | ✗ |
| Cone of Influence Masking | Time-Frequency | CWT | ★★★☆☆ | ✗ (quality filter) |
| Inter-Scale Correlation | Cross-Scale | DWT / CWT | ★★★☆☆ | ✗ |
| WPD Best-Basis Energy Distribution | Cross-Scale | WPD | ★★★☆☆ | ✗ |
| Wavelet Coherence | Cross-Channel | CWT | ★★★★★ | ✗ |
| Wavelet PLV (wPLV) | Cross-Channel | CWT | ★★★★☆ | ✗ |
| Wavelet PAC (wPAC) | Cross-Channel | CWT | ★★★★★ | ✗ |
