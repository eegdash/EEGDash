r"""Avalanche feature extraction pipeline.

Basic avalanche detection and binning functions:
- avalanche_preprocessor: detect avalanche events in data
- bin_avalanches: bin avalanche events into contiguous time bins
- detect_avalanches: detect avalanche start and end indices in binned array

"""
import eegdash.features as feat
from eegdash.features.feature_bank.signal import SIGNAL_PREDECESSORS

import warnings
import numpy as np
import os

import scipy.ndimage as ndi
from scipy.stats import median_abs_deviation
from scipy.optimize import minimize_scalar

EPSILON = 1e-10

# ============= Utils =============

def _nll(exponent: float, 
         x_vals_logs: np.ndarray, 
         sum_log_data: float, 
         n: int) -> float:
    r"""
    Calculate the Negative Log-Likelihood (NLL) for a discrete truncated power-law distribution.

    The NLL is derived from the probability mass function (PMF):
    $$P(x) = \frac{x^{-\gamma}}{Z(\gamma, x_{min}, x_{max})}$$
    where $Z$ is the transcendental Hurwitz zeta-like normalization constant:
    $$Z(\gamma, x_{min}, x_{max}) = \sum_{k=x_{min}}^{x_{max}} k^{-\gamma}$$

    The objective function to minimize is:
    $$\mathcal{L}(\gamma) = \gamma \sum_{i=1}^{n} \ln(x_i) + n \ln \left( \sum_{k=x_{min}}^{x_{max}} k^{-\gamma} \right)$$

    Parameters
    ----------
    exponent : float
        The power-law exponent ($\gamma$) being optimized.
    x_vals_logs : np.ndarray
        Logarithm of the discrete x values in the fitting range.
    sum_log_data : float
        The sum of the logarithm of the observed data points in the fitting range.
    n : int
        The number of data points in the fitting range.

    Returns
    -------
    float
        The negative log-likelihood value. Returns infinity if the exponent $\le 1$ 
        or if the normalization factor $Z$ is non-finite.
"""
    if exponent <= 1: 
        return np.inf
    Z = np.sum(np.exp(-exponent * x_vals_logs)) # np.sum(x_vals ** (-exponent))
    if Z <= 0 or not np.isfinite(Z): 
        return np.inf
    return exponent * sum_log_data + n * np.log(Z)

def _fit_truncated_power_law(data: np.ndarray, 
                             system_size: int, 
                             n_min: int = 500, 
                             cutoff_search_step: int = 1,
                             xmin: int = 1) -> dict:
    r"""
    Estimates the power-law exponent using a Truncated Maximum Likelihood Estimation (MLE) 
    combined with Kolmogorov-Smirnov (KS) distance minimization.

    This method follows the "Fekete-style" Algorithm, which adapts the 
    Clauset et al. (2009) framework to handle finite-size effects in biological systems. 
    It performs a grid search over potential window boundaries [xmin, xmax] to identify 
    the optimal range where the data most closely follows a power-law distribution.

    Parameters
    ----------
    data : np.ndarray
        A 1-D array of avalanche metrics (e.g., sizes or durations). 
    system_size : int, optional
        The physical limit of the recording system (e.g., the total number of channels). 
        If None, the search range is capped at the maximum observed value in the data.
    n_min : int, default=500
        The minimum number of data points required within a [xmin, cutoff] window 
        to consider a fit statistically reliable.
    cutoff_search_step : int, default=1
        The step size for searching cutoff candidates.
    xmin : int, default=1
        The lower bound of the fitting range. Must be >= 1 for discrete power-law.

    Returns
    -------
    results : dict
        A dictionary containing the parameters of the best-fitting truncated power law:
        
        * 'exponent' (float): The estimated power-law index. It represents the probability 
          density function slope $P(x) \propto x^{-exponent}$.
        * 'xmin' (float): The lower bound of the optimal fitting window.
        * 'cutoff' (float): The upper bound of the optimal fitting window.
        * 'ks' (float): The Kolmogorov-Smirnov distance. Lower values indicate a better fit.
        * 'n_included' (int): The number of individual avalanche events contained 
          within the selected [xmin, cutoff] range.

    """
    assert system_size != None and system_size > 0, "system_size must be a positive integer."
    
    # Data preparation
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data >= 1]
    
    if data.size < n_min:
        return {'exponent': np.nan, 
                'xmin': np.nan, 
                'cutoff': np.nan, 
                'ks': np.nan, 
                'n_included': 0}

    data = np.sort(data)
    maxdata = data[-1]

    # Cutoff definition
    lower_bound = int(0.8 * system_size)
    upper_bound = int(1.5 * system_size)
    cutoff_candidates = range(lower_bound, upper_bound + 1, cutoff_search_step)
    cutoff_valid = [_ for _ in cutoff_candidates if _ <= maxdata] 
    if not cutoff_valid:
        cutoff_valid = [int(maxdata)]

    # --- Grid Search (Minimize KS) ---
    best_ks = np.inf
    best_params = {'exponent': np.nan, 
                   'xmin': np.nan, 
                   'cutoff': np.nan, 
                   'ks': np.nan, 
                   'n_included': 0}
        
    start_idx = np.searchsorted(data, xmin, side='left')
    log_data_cumsum = np.insert(np.cumsum(np.log(data)), 0, 0.0)
    
    for cutoff in cutoff_valid:
        end_idx = np.searchsorted(data, cutoff, side='right')
        n = end_idx - start_idx        
        if n < n_min:
            continue

        # --- MLE Fit ---
        x_vals = np.arange(xmin, cutoff + 1)
        x_vals_logs = np.log(x_vals)
        sum_log_data = log_data_cumsum[end_idx] - log_data_cumsum[start_idx]

        res = minimize_scalar(_nll, 
                              bounds=(1.0001, 10), 
                              method="bounded",
                              args=(x_vals_logs, sum_log_data, n))
        exponent = float(res.x)

        # --- KS Distance Calculation ---

        # Theoretical CDF
        pdf_theory = np.exp(-exponent * x_vals_logs) # x_vals ** (-exponent)
        pdf_theory /= pdf_theory.sum() # Normalize
        cdf_theory = np.cumsum(pdf_theory)
        
        # Empirical CDF
        window_slice = data[start_idx:end_idx]
        cdf_emp = np.searchsorted(window_slice, x_vals, side="right") / n

        # Compute KS
        ks = np.max(np.abs(cdf_emp - cdf_theory))

        # --- Update Best Fit ---
        if ks < best_ks:
            best_ks = ks
            best_params = {
                'exponent': exponent,
                'xmin': xmin,
                'cutoff': cutoff,
                'ks': ks,
                'n_included': n
            }
    
    return best_params

# ============= Preprocessors =============

@feat.FeaturePredecessor()
def avalanche_preprocessor(data: np.ndarray, 
                           fs: float, 
                           k: float=3.0, 
                           thresholding: str='std',
                           save_dir: str = None,
                           save_prefix: str = "") -> tuple[np.ndarray, float]:
    r"""
    Detect avalanche events in the data.

    Params
    ------
    data : np.ndarray
        Input data array of size (n_windows, n_channels, n_samples).
        if 2D (n_channels, n_samples) is provided, it will be treated as a single window.
    fs : float
        Sampling frequency of the data in Hz.
    k : float
        Threshold multiplier for standard deviation.
    thresholding : str
        Method for thresholding ('std' or 'mad').
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    binary_data : np.ndarray (uint8)
        Binary array of shape (n_windows, n_channels, n_samples), with 1s at 
        the absolute peak of each detected event.
    fs : float
        Sampling frequency (passed through).
    """
    original_ndim = data.ndim
    if original_ndim == 2:
        data = data[np.newaxis, :, :] # (1, channels, time)
    elif original_ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
    
    # compute absolute normalized data
    if thresholding == 'std':
        row_means = data.mean(axis=2, keepdims=True)
        row_stds = data.std(axis=2, keepdims=True)
        row_stds[row_stds == 0] = EPSILON
        abs_data = data - row_means 
        abs_data /= row_stds         
        np.abs(abs_data, out=abs_data)

    elif thresholding == 'mad':
        row_medians = np.median(data, axis=2, keepdims=True)
        robust_sd = median_abs_deviation(data, axis=2, scale='normal', keepdims=True)
        robust_sd[robust_sd == 0] = EPSILON
        abs_data = data - row_medians 
        abs_data /= robust_sd         
        np.abs(abs_data, out=abs_data)

    else:
        raise ValueError(f"Unsupported thresholding method: {thresholding}")
    
    # create mask and structure for ndimage
    mask = (abs_data > k).astype(np.uint8)
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1 # connect along time axis only

    # label theshold-crossing events
    labels, num_features = ndi.label(mask, structure=structure)

    binary_data = np.zeros_like(data, dtype=np.uint8)

    if num_features != 0:
        max_positions = ndi.maximum_position(abs_data, labels, index=np.arange(1, num_features + 1))
        epochs, rows, cols = zip(*max_positions)
        binary_data[epochs, rows, cols] = 1

    # save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_avalanche_preprocessor.npz")
        np.savez_compressed(
            filename,
            data=binary_data,
            fs=fs,
            multiplier=k,
            thresholding_method=thresholding,
            function="avalanche_preprocessor"
        )
        print(f"Saved {filename}")

    return binary_data, fs

@feat.FeaturePredecessor(avalanche_preprocessor)
def bin_avalanches(binary_data: np.ndarray, 
                   fs: float, 
                   bin_size_sec: float = None,
                   bin_size_samples: int = None,
                   n_bins: int = None,
                   save_dir: str = None,
                   save_prefix: str = "") -> tuple[np.ndarray, float, float, int]:
    r"""
    Bin avalanche events into contiguous time bins.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_windows, n_channels, n_samples) indicating avalanche events.
        if 2D (n_channels, n_samples) is provided, it will be treated as a single window.
    fs : float
        Sampling frequency of the data in Hz.
    bin_size_sec : float, optional
        Desired bin size in seconds. If specified, overrides other binning parameters.
    bin_size_samples : int, optional
        Desired bin size in samples. If specified, overrides n_bins.
    n_bins : int, optional
        Desired number of bins to divide the data into. If specified, overrides bin_size_sec.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    binned_array : np.ndarray
        2D ndarray of shape (n_epochs, n_bins) of binned avalanche events.
    fs : float
        Sampling frequency (passed through).
    bin_size_sec : float
        The bin size used in seconds.
    actual_n_bins : int
        The number of bins used after trimming the data.

    Notes
    -----
    - Exactly one of bin_size_sec, bin_size_samples, or n_bins can be specified. 
    - The function will trim the data to fit an integer number of bins if necessary.
    """  
    # reshape to (n_windows, n_channels, n_samples) if needed
    original_ndim = binary_data.ndim
    if original_ndim == 2:
        binary_data = binary_data[np.newaxis, :, :]
    elif original_ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got {binary_data.ndim}D")
    
    # verify that only one of the binning parameters is specified
    preds = [bin_size_sec != None, bin_size_samples != None, n_bins != None]
    if sum(preds) > 1:
        raise ValueError("Only one of bin_size_sec, bin_size_samples, or n_bins can be specified.")
    elif sum(preds) == 0:
        raise ValueError("One of bin_size_sec, bin_size_samples, or n_bins must be specified.") 
    
    if bin_size_sec:
        bin_size_samples = int(np.floor(bin_size_sec * fs))
    elif n_bins:
        bin_size_samples = binary_data.shape[2] // n_bins

    if bin_size_samples <= 1:
        raise ValueError("Resulting bin size is too small")
    
    bin_size_sec = bin_size_samples / fs # recalculate
    
    network_activity = np.sum(binary_data, axis=1) # sum across channels
    n_epochs, n_samples = network_activity.shape

    actual_n_bins = n_samples // bin_size_samples

    trimmed_activity = network_activity[:, :actual_n_bins * bin_size_samples] # trim to fit bins
    binned_array = trimmed_activity.reshape(n_epochs, actual_n_bins, bin_size_samples).sum(axis=2)

    # save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_bin_avalanches.npz")
        np.savez_compressed(
            filename,
            data=binned_array,
            fs=fs,
            bin_size_sec=bin_size_sec,
            bin_size_samples=bin_size_samples,
            n_bins=actual_n_bins,
            function="bin_avalanches"
        )
        print(f"Saved {filename}")

    return binned_array, fs, bin_size_sec, actual_n_bins

@feat.FeaturePredecessor(bin_avalanches)
def detect_avalanches(binned_array: np.ndarray, 
                      fs: float, 
                      bin_size_sec: float,
                      n_bins: int,
                      save_dir: str = None,
                      save_prefix: str = "") -> dict:
    r"""
    Detect avalanche start and end indices in the binned array.

    Params
    ------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events. If 2D (n_epochs, n_bins) is provided, 
        it will process each epoch separately.
    fs : float
        Sampling frequency of the data in Hz.
    bin_size_sec : float
        The bin size used in seconds.
    n_bins : int
        The number of bins in the binned array.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    ------- 
    avalanche_dict : dict
        Dictionary with keys:
        - 'fs': float, the sampling frequency (passed through).
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.
        - 'avalanches': list of dicts, each with keys:
            - 'data': 1D array of binned activity for the epoch.
            - 'indices': 2D array of shape (n_avalanches, 2) with start and end bin indices.
            - 'epoch': int, the epoch index.
    """
    # reshape to (n_epochs, n_bins) if needed
    original_ndim = binned_array.ndim
    if original_ndim == 1:
        binned_array = binned_array[np.newaxis, :]

    n_epochs = binned_array.shape[0]
    avalanche_dict = {
        'fs': fs,
        'bin_size_sec': bin_size_sec,
        'n_bins': n_bins,
        'avalanches': []
    }
    
    for epoch in range(n_epochs):
        epoch_binned = binned_array[epoch]
        is_active = (epoch_binned > 0).astype(int)
        n = len(is_active)

        if n == 0 or not np.any(is_active):
            avalanche_dict['avalanches'].append({
                'data': epoch_binned,
                'indices': np.empty((0, 2), dtype=int),
                'epoch': epoch,   
            })
            continue

        diffs = np.diff(is_active, prepend=0, append=0) # pad to detect edges

        start_indices = np.where(diffs == 1)[0]
        end_indices = np.where(diffs == -1)[0] - 1 # end is inclusive

        # Filter edge cases
        if len(start_indices) > 0:
            if start_indices[0] == 0:
                start_indices = start_indices[1:]
                end_indices = end_indices[1:]
            if len(end_indices) > 0:
                last_bin_idx = len(epoch_binned) - 1
                if end_indices[-1] == last_bin_idx:
                    start_indices = start_indices[:-1]
                    end_indices = end_indices[:-1]

        avalanche_dict['avalanches'].append({
            'data': epoch_binned,
            'indices': np.column_stack((start_indices, end_indices)),
            'epoch': epoch,
        })

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_detect_avalanches.npz")
        np.savez_compressed(
            filename,
            fs=fs,
            bin_size_sec=bin_size_sec,
            n_bins=n_bins,
            data=avalanche_dict['avalanches'],
            function="detect_avalanches"
        )
        print(f"Saved {filename}")

    return avalanche_dict

# ============= Features =============

@feat.FeaturePredecessor(detect_avalanches)
@feat.multivariate_feature
def branching_parameter(avalanche_dict: dict, 
                        method: str = 'naive',
                        save_dir: str = None,
                        save_prefix: str = "") -> float:
    r"""
    Calculate the branching parameter of avalanche events.
    
    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'fs': float, the sampling frequency (passed through).
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.
        - 'avalanches': list of dicts, each with keys:
            - 'data': 1D array of binned activity for the epoch.
            - 'indices': 2D array of shape (n_avalanches, 2) with start and end bin indices.
            - 'epoch': int, the epoch index.
    method : str, optional
        Method to calculate branching parameter:
        * 'naive': Mean of ratios.
        * 'weighted': Ratio of sums.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    sigmas : list of float
        Branching parameter per epoch. Returns np.nan for epochs with no avalanches.
    
    Raises
    ------
    ValueError
        If an unsupported method is provided.
    
    Notes
    -----
    Based on Beggs, John M., and Dietmar Plenz. "Neuronal avalanches in neocortical circuits" (2003).
    
    * Naive: Treats every avalanche as an equal statistical event.
    * Weighted: Treats every active electrode (ancestor) as a statistical event.

    """
    
    avalanches = avalanche_dict.get('avalanches', [])
    n_epochs = len(avalanches)

    sigmas = np.full(n_epochs, np.nan, dtype=float)
    if save_dir:
        n_a_list = []
        n_d_list = []
    
    for i, ep in enumerate(avalanches):
        binned_array = ep['data']
        indices = ep['indices']

        if indices.shape[0] == 0:
            if save_dir:
                n_a_list.append(np.array([]))
                n_d_list.append(np.array([]))
            continue
            
        starts = indices[:, 0]
        n_a = binned_array[starts].astype(np.float64)
        n_d = binned_array[starts + 1].astype(np.float64)

        if method == 'naive':
            sigma = np.mean(n_d / n_a)

        elif method == 'weighted':
            # TODO: check if this is correct
            sigma = np.sum(n_d) / np.sum(n_a)
            warnings.warn("Weighted method wasn't verified.")

        else:
            raise ValueError(f"Unsupported method: {method}")
        
        sigmas[i] = float(sigma)

        if save_dir:
            n_a_list.append(n_a)
            n_d_list.append(n_d)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_branching_parameter.npz")
        n_a_arr = np.empty(len(n_a_list), dtype=object)
        n_a_arr[:] = n_a_list
        n_d_arr = np.empty(len(n_d_list), dtype=object)
        n_d_arr[:] = n_d_list
        np.savez_compressed(
            filename,
            n_a=n_a_arr,
            n_d=n_d_arr,
            data=sigmas,
            method=method,
            function="branching_parameter"
        )
        print(f"Saved {filename}")
    
    return sigmas

@feat.FeaturePredecessor(detect_avalanches)
@feat.multivariate_feature
def alpha_exponent(avalanche_dict: dict, 
                   n_channels: int = None,
                   ks_threshold: float = 0.1,
                   save_dir: str = None,
                   save_prefix: str = "") -> np.ndarray:
    r"""
    Calculate the Alpha exponent of avalanche size distribution.

    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'fs': float, the sampling frequency (passed through).
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.
        - 'avalanches': list of dicts, each with keys:
            - 'data': 1D array of binned activity for the epoch.
            - 'indices': 2D array of shape (n_avalanches, 2) with start and end bin indices.
            - 'epoch': int, the epoch index.
    n_channels : int, optional
        The physical limit of the recording system (e.g., the total number of channels).
    ks_threshold : float, optional
        Threshold for the Kolmogorov-Smirnov distance to consider the fit reliable. 
        If the best fit has a KS distance above this threshold, the function will return np.nan.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    alphas : np.ndarray
        Alpha exponents of the avalanche size distribution per epoch.
        Returns np.nan for epochs with no avalanches or if the fit is unreliable.
    """
    avalanches = avalanche_dict.get('avalanches', [])
    n_epochs = len(avalanches)

    if n_channels is None:
        warnings.warn("n_channels not provided. Alpha exponent fit may be unreliable.")

    alphas = np.full(n_epochs, np.nan, dtype=float)

    if save_dir:
        sizes_list = []
        xmin_list = np.full(n_epochs, np.nan, dtype=float)
        cutoff_list = np.full(n_epochs, np.nan, dtype=float)
        n_included_list = np.full(n_epochs, np.nan, dtype=float)
        fit_ks_list = np.full(n_epochs, np.nan, dtype=float)

    for i, ep in enumerate(avalanches):
        binned_array = ep['data']
        indices = ep['indices']

        if indices.shape[0] == 0:
            if save_dir:
                sizes_list.append(np.array([]))
            continue
        
        sizes = np.add.reduceat(binned_array, indices[:, 0]) # C-level array operation

        if n_channels is None:
            system_size = np.max(sizes) if sizes.size > 0 else None
            fit_results = _fit_truncated_power_law(sizes, system_size=system_size)
        else:
            fit_results = _fit_truncated_power_law(sizes, system_size=n_channels)

        if fit_results['ks'] <= ks_threshold:
            alphas[i] = fit_results['exponent']

        if save_dir:
            sizes_list.append(sizes)
            xmin_list[i] = fit_results.get('xmin', np.nan)
            cutoff_list[i] = fit_results.get('cutoff', np.nan)
            n_included_list[i] = fit_results.get('n_included', np.nan)
            fit_ks_list[i] = fit_results.get('ks', np.nan)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_alpha_exponent.npz")
        
        sizes_array = np.empty(len(sizes_list), dtype=object)
        sizes_array[:] = sizes_list
        
        np.savez_compressed(
            filename,
            data=alphas,
            sizes=sizes_array,
            xmin=xmin_list,
            cutoff=cutoff_list,
            n_includeds=n_included_list,
            fit_kss=fit_ks_list,
            ks_threshold=ks_threshold,
            n_channels=n_channels if n_channels is not None else -1,
            function="alpha_exponent"
        )
        print(f"Saved {filename}")
    
    return alphas

@feat.FeaturePredecessor(detect_avalanches)
@feat.multivariate_feature
def tau_exponent(avalanche_dict: dict, 
                 t_max_method: str = 'max',
                 ks_threshold: float = 0.1,
                 save_dir: str = None,
                 save_prefix: str = "") -> np.ndarray:
    r"""
    Calculate the tau exponent of avalanche duration distribution.

    Note: The fitting is performed on the discrete bin counts (integers) to satisfy 
    the Discrete MLE assumptions. The resulting exponent 'tau' is scale-invariant 
    and valid for physical time units as well.

    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'fs': float, the sampling frequency (passed through).
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.
        - 'avalanches': list of dicts, each with keys:
            - 'data': 1D array of binned activity for the epoch.
            - 'indices': 2D array of shape (n_avalanches, 2) with start and end bin indices.
            - 'epoch': int, the epoch index.
    t_max_method : str, optional
        Method to determine the maximum duration (t_max) for fitting:
        * 'max': Use the maximum observed duration in the data.
        * 'lab': Use the theoretical t_max based on the maximum avalanche size and average activity.
    ks_threshold : float, optional
        Threshold for the Kolmogorov-Smirnov distance to consider the fit reliable. 
        If the best fit has a KS distance above this threshold, the function will return np.nan.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    taus : ndarray
        Tau exponents of the avalanche duration distribution per epoch. 
        Returns np.nan for epochs with no avalanches or if the fit is unreliable.
    """
    
    avalanches = avalanche_dict.get('avalanches', [])
    n_epochs = len(avalanches)

    taus = np.full(n_epochs, np.nan, dtype=float)

    if save_dir:
        durations_list = []
        xmin_list = np.full(n_epochs, np.nan, dtype=float)
        cutoff_list = np.full(n_epochs, np.nan, dtype=float)
        n_included_list = np.full(n_epochs, np.nan, dtype=float)
        fit_ks_list = np.full(n_epochs, np.nan, dtype=float)

    for i, ep in enumerate(avalanches):
        indices = ep['indices']

        if indices.shape[0] == 0:
            if save_dir:
                durations_list.append(np.array([]))
            continue

        durations_bins = indices[:, 1] - indices[:, 0] + 1
        
        if t_max_method == 'max':
            t_max = np.max(durations_bins)
        elif t_max_method == 'lab':
            # Compute theoretical t_max based on maximun avalanche size
            binned_array = ep['data']
            n_bins = avalanche_dict['n_bins']
            max_size = np.max(np.add.reduceat(binned_array, indices[:, 0]))
            coeff = np.sum(binned_array) / n_bins
            t_max = int(np.sqrt(max_size / coeff))
        else:
            raise ValueError(f"Unsupported t_max_method: {t_max_method}")
                
        fit_results = _fit_truncated_power_law(durations_bins, system_size=t_max)

        if fit_results['ks'] <= ks_threshold:
            taus[i] = fit_results['exponent']

        if save_dir:
            durations_list.append(durations_bins)
            xmin_list[i] = fit_results.get('xmin', np.nan)
            cutoff_list[i] = fit_results.get('cutoff', np.nan)
            n_included_list[i] = fit_results.get('n_included', np.nan)
            fit_ks_list[i] = fit_results.get('ks', np.nan)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_tau_exponent.npz")
        
        durations_array = np.empty(len(durations_list), dtype=object)
        durations_array[:] = durations_list

        np.savez_compressed(
            filename,
            data=taus,
            durations=durations_array,
            xmin=xmin_list,
            cutoff=cutoff_list,
            n_includeds=n_included_list,
            fit_kss=fit_ks_list,
            t_max_method=t_max_method,
            ks_threshold=ks_threshold,
            function="tau_exponent"
        )
        print(f"Saved {filename}")
    
    return taus

@feat.FeaturePredecessor(detect_avalanches)
@feat.multivariate_feature
def gamma_exponent(avalanche_dict: dict,
                   min_unique_durations: int = 3,
                   mse_threshold: float = 0.05,
                   save_dir: str = None,
                   save_prefix: str = "") -> np.ndarray:
    r"""
    Estimate the Gamma exponent of the scaling relationship between avalanche size and duration.
    
    Params
    ------
    avalanche_dict : dict
        Dictionary with keys:
        - 'fs': float, the sampling frequency (passed through).
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.
        - 'avalanches': list of dicts, each with keys:
            - 'data': 1D array of binned activity for the epoch.
            - 'indices': 2D array of shape (n_avalanches, 2) with start and end bin indices.
            - 'epoch': int, the epoch index.
    min_unique_durations : int, optional
        Minimum number of unique durations required to perform the fit.
    mse_threshold : float, optional
        Maximum mean squared error (MSE) of the log-log linear fit to consider the Gamma exponent reliable. 
        If the MSE exceeds this threshold, the function will return np.nan for that epoch.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    gamma_obs : np.ndarray
        Gamma exponents of the scaling relationship between avalanche size and duration per epoch.
        Returns np.nan if not enough avalanches, or if the fit fails.
    
    Notes
    -----
    This is a scaling relationship, and not a probability distribution fit. Therefore, use
    the continuous power-law fitting approach (log-log linear regression), instead of the
    discrete MLE approach used for alpha and tau.

    Assertions
    ----------
    - At least 2 unique durations are required to fit the scaling relationship. 
    """
    
    assert min_unique_durations >= 2, \
        "At least 2 unique durations are required to fit the scaling relationship."
    
    avalanches = avalanche_dict.get('avalanches', [])
    n_epochs = len(avalanches)

    gamma_obs = np.full(n_epochs, np.nan, dtype=float)

    if save_dir:
        intercepts = np.full(n_epochs, np.nan, dtype=float)
        unique_durations_list = []
        avg_sizes_list = []

    for i, ep in enumerate(avalanches):

        indices = ep['indices']
        binned_array = ep['data']

        if indices.shape[0] == 0:
            if save_dir:
                unique_durations_list.append(np.array([]))
                avg_sizes_list.append(np.array([]))
            continue

        sizes = np.add.reduceat(binned_array, indices[:, 0])
        durations = indices[:, 1] - indices[:, 0] + 1
        unique_durations = np.unique(durations)

        if len(unique_durations) < min_unique_durations:
            if save_dir:
                unique_durations_list.append(np.array([]))
                avg_sizes_list.append(np.array([]))
            continue
        
        avg_sizes = []
        for t in unique_durations:
            avg_sizes.append(np.mean(sizes[durations == t]))

        avg_sizes = np.array(avg_sizes)

        log_t = np.log10(unique_durations)
        log_s = np.log10(avg_sizes)

        poly_results = np.polyfit(log_t, log_s, 1, full=True)
        residuals = poly_results[1]

        if residuals.size > 0:
            mse = residuals[0] / len(unique_durations)
            if mse <= mse_threshold:
                gamma_obs[i] = poly_results[0][0]
                if save_dir:
                    intercepts[i] = poly_results[0][1]
        elif residuals.size == 0:
            gamma_obs[i] = poly_results[0][0]
            if save_dir:
                intercepts[i] = poly_results[0][1]

        if save_dir:
            if not np.isnan(gamma_obs[i]):
                unique_durations_list.append(unique_durations)
                avg_sizes_list.append(avg_sizes)
            else:
                unique_durations_list.append(np.array([]))
                avg_sizes_list.append(np.array([]))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_gamma_exponent.npz")
        
        ud_array = np.empty(len(unique_durations_list), dtype=object)
        ud_array[:] = unique_durations_list
        
        as_array = np.empty(len(avg_sizes_list), dtype=object)
        as_array[:] = avg_sizes_list

        np.savez_compressed(
            filename,
            data=gamma_obs,
            intercepts=intercepts,
            unique_durations=ud_array,
            avg_sizes=as_array,
            min_unique_durations=min_unique_durations,
            mse_threshold=mse_threshold,
            function="gamma_exponent"
        )
        print(f"Saved {filename}")
    
    return gamma_obs

# TODO: this will not work
@feat.FeaturePredecessor(alpha_exponent, tau_exponent, gamma_exponent)
@feat.multivariate_feature
def dcc(alphas: np.ndarray, 
        taus: np.ndarray,
        gammas: np.ndarray,
        save_dir: str = None,
        save_prefix: str = "") -> np.ndarray:
    r"""
    Calculate the Deviation from Criticality Coefficient (DCC) based on the observed exponents:

    $$ DCC = | \gamma_{obs} - \gamma_{pred} | $$

    Where: $$ \gamma_{pred} = \frac{\tau - 1}{\alpha - 1} $$

    Params
    ------
    alphas : np.ndarray
        Alpha exponents of the avalanche size distribution per epoch.
    taus : np.ndarray
        Tau exponents of the avalanche duration distribution per epoch.
    gammas : np.ndarray
        Gamma exponents of the scaling relationship between avalanche size and duration per epoch.
    save_dir : str, optional
        Directory to save results. If None (default), results won't be saved.
    save_prefix : str, optional
        Prefix for saved result files.

    Returns
    -------
    dcc_values : np.ndarray
        Deviation from Criticality Coefficient (DCC) for each epoch. A value close to 0 indicates criticality.

    Assertions
    ----------
    - All input arrays must have the same shape.

    """
    assert alphas.shape == taus.shape == gammas.shape, \
        "All input arrays must have the same shape."

    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_pred = (taus - 1.0) / (alphas - 1.0)
        dcc_values = np.abs(gammas - gamma_pred)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{save_prefix}_dcc.npz")
        np.savez_compressed(
            filename,
            data=dcc_values,
            alphas=alphas,
            taus=taus,
            gammas=gammas,
            function="dcc"
        )
        print(f"Saved {filename}")
    
    return dcc_values
