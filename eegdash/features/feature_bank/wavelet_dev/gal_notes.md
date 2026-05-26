# Notes for further implementation

1. the wavelet's PLV should have two predecessor options - the wavelet preprocessor and the analytic signal pp. (note that the analytic signal is inside 'connectivity' - need to pull from github).

2. the leading axis of the returned arrays shape should always be n_batches (batch size) - as in the other package's preprocessors.

3. the 'wavelet_connectivity_preprocessor' start with 3 lines identicle to 'wavelet_preprocessor'. we shouldchange this: wavelet_preprocessor should be the predecessor of wavelet_connectivity_preprocessor. implementing this will cause '_compute_cwt' to be called only once in the script - so we should remove this helper and integrate into the preprocessor.

4. never perform 'np.abs(z)**2' on complex z. instead use: z.real^2+z.imag^2. its faster.

5. if possible, decorate with numba njit. funcs like 'wavelet_pac' include double for loops, we should do whatever it takes to extract some of this code to numba.

6. in 'wavelet_plv' - dont use 'np.angle'. instead, we can divide Wx by |Wx| and Wy by |Wy| and multiply the x result by the conj of Wy result. than proceed to mean, abs.


# Notes for my understanding 

1. why do we need the connectiviry preprocessor?
2. why do we need to compute central freqs and central freq of the band?