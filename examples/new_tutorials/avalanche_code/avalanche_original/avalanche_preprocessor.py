r"""Avalanche preprocessing and binning functions.

Basic avalanche detection and binning functions:
- avalanche_preprocessor: detect avalanche events in data
- bin_avalanches: bin avalanche events into contiguous time bins
- detect_avalanches: detect avalanche start and end indices in binned array
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import median_abs_deviation

EPSILON = 1e-10

def avalanche_preprocessor(data: np.ndarray, 
                           fs: float, 
                           k: float=3.0, 
                           thresholding: str='std') -> tuple[np.ndarray, float]:
    r"""
    Detect avalanche events in the data.

    Params
    ------
    data : np.ndarray
        Input data array of size (n_channels, n_samples).
    fs : float
        Sampling frequency of the data in Hz.
    k : float
        Threshold multiplier for standard deviation.
    thresholding : str
        Method for thresholding ('std' or 'mad').
    
    Returns
    -------
    binary_data : np.ndarray (uint8)
        Binary array of the same shape as data, with 1s at 
        the absolute peak of each detected event.
    fs : float
        Sampling frequency (passed through).
    """
    # compute absolute normalized data
    if thresholding == 'std':
        row_means = data.mean(axis=1, keepdims=True)
        row_stds = data.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = EPSILON
        abs_data = data - row_means 
        abs_data /= row_stds         
        np.abs(abs_data, out=abs_data)

    elif thresholding == 'mad':
        row_medians = np.median(data, axis=1, keepdims=True)
        robust_sd = median_abs_deviation(data, axis=1, scale='normal', keepdims=True)
        robust_sd[robust_sd == 0] = EPSILON
        abs_data = data - row_medians 
        abs_data /= robust_sd         
        np.abs(abs_data, out=abs_data)

    else:
        raise ValueError(f"Unsupported thresholding method: {thresholding}")
    
    # create mask and structure for ndimage
    mask = (abs_data > k).astype(np.uint8)
    structure = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]])

    # label theshold-crossing events
    labels, num_features = ndi.label(mask, structure=structure)

    binary_data = np.zeros_like(data, dtype=np.uint8)

    if num_features != 0:
        max_positions = ndi.maximum_position(abs_data, labels, index=np.arange(1, num_features + 1))
        rows, cols = zip(*max_positions)
        binary_data[rows, cols] = 1

    return binary_data, fs

def bin_avalanches(binary_data: np.ndarray, 
                   fs: float, 
                   bin_size_sec: float = None,
                   bin_size_samples: int = None,
                   n_bins: int = None) -> tuple[np.ndarray, float, float, int]:
    r"""
    Bin avalanche events into contiguous time bins.

    Params
    ------
    binary_data : np.ndarray
        Binary array of size (n_channels, n_samples) indicating avalanche events.
    fs : float
        Sampling frequency of the data in Hz.
    bin_size_sec : float, optional
        Desired bin size in seconds. If specified, overrides other binning parameters.
    bin_size_samples : int, optional
        Desired bin size in samples. If specified, overrides n_bins.
    n_bins : int, optional
        Desired number of bins to divide the data into. If specified, overrides bin_size_sec.

    Returns
    -------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
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
    # verify that only one of the binning parameters is specified
    preds = [bin_size_sec != None, bin_size_samples != None, n_bins != None]
    if sum(preds) > 1:
        raise ValueError("Only one of bin_size_sec, bin_size_samples, or n_bins can be specified.")
    elif sum(preds) == 0:
        raise ValueError("One of bin_size_sec, bin_size_samples, or n_bins must be specified.") 
    
    if bin_size_sec:
        bin_size_samples = int(np.floor(bin_size_sec * fs))
    elif n_bins:
        bin_size_samples = binary_data.shape[1] // n_bins

    if bin_size_samples <= 1:
        raise ValueError("Resulting bin size is too small")
    
    bin_size_sec = bin_size_samples / fs # recalculate

    network_activity = np.sum(binary_data, axis=0)
    n_samples = network_activity.shape[0]
    actual_n_bins = n_samples // bin_size_samples
    trimmed_activity = network_activity[:actual_n_bins * bin_size_samples] # trim to fit bins

    binned_array = trimmed_activity.reshape(actual_n_bins, bin_size_samples).sum(axis=1)

    return binned_array, fs, bin_size_sec, actual_n_bins

def detect_avalanches(binned_array: np.ndarray, 
                      fs: float, 
                      bin_size_sec: float,
                      n_bins: int) -> dict:
    r"""
    Detect avalanche start and end indices in the binned array.

    Params
    ------
    binned_array : np.ndarray
        1-D ndarray of binned avalanche events.
    bin_size_sec : float
        The bin size used in seconds.

    Returns
    ------- 
    avalanche_dict : dict
        Dictionary with keys:
        - 'data': original binned_array
        - 'fs': float, the sampling frequency (passed through).
        - 'indices': np.ndarray of shape (n_avalanches, 2) with start and end indices of each avalanche.
            the end index is inclusive.
        - 'bin_size_sec': float, the bin size used (in seconds).
        - 'n_bins': int, the number of bins in the binned array.

    """
    is_active = (binned_array > 0).astype(int)
    n = len(is_active)

    if n == 0 or not np.any(is_active):
        return {
            'data': binned_array,
            'fs': fs,
            'indices': np.empty((0, 2), dtype=int),
            'bin_size_sec': bin_size_sec,
            'n_bins': n_bins
        }   
    
    diffs = np.diff(is_active, prepend=0, append=0) # pad to detect edges

    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0] - 1 # end is inclusive

    # Filter edge cases
    if len(start_indices) > 0:
        if start_indices[0] == 0:
            start_indices = start_indices[1:]
            end_indices = end_indices[1:]
        if len(end_indices) > 0:
            last_bin_idx = len(binned_array) - 1
            if end_indices[-1] == last_bin_idx:
                start_indices = start_indices[:-1]
                end_indices = end_indices[:-1]

    return {
        'data': binned_array,
        'fs': fs,
        'indices': np.column_stack((start_indices, end_indices)),
        'bin_size_sec': bin_size_sec,
        'n_bins': n_bins
    }

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate synthetic data
    np.random.seed(0)
    n_channels = 5
    n_samples = 10000
    fs = 100.0
    data = np.random.randn(n_channels, n_samples)

    # preprocess and bin avalanches
    binary_data, fs = avalanche_preprocessor(data, fs, k=2.0, thresholding='std')
    binned_array, fs, bin_size_sec, n_bins = bin_avalanches(binary_data, fs, bin_size_sec=0.1)
    avalanche_dict = detect_avalanches(binned_array, fs, bin_size_sec, n_bins)

    print("Avalanche indices (start_bin, end_bin):")
    print(avalanche_dict['indices'])

    # plot binned activity and detected avalanches
    plt.figure(figsize=(12, 4))
    plt.plot(avalanche_dict['data'], label='Binned Activity')
    for start_bin, end_bin in avalanche_dict['indices']:
        plt.axvspan(start_bin, end_bin + 1, color='red', alpha=0.3)
    plt.xlabel('Bin Index')
    plt.ylabel('Activity Count')
    plt.title('Binned Avalanche Activity with Detected Avalanches')
    plt.legend()
    plt.show()