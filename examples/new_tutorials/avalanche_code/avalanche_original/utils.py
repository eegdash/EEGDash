r"""Utility functions for avalanche analysis.

- _nll: negative log-likelihood function for discrete truncated power-law
- _fit_truncated_power_law: fit truncated power-law to avalanche data
"""
import numpy as np
from scipy.optimize import minimize_scalar

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
