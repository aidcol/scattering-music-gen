from typing import List, Dict, Any, Tuple, Optional

from IPython.display import Audio
import math
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
from torch import Tensor


def normalize_audio(audio: np.ndarray, eps: float = 1e-10):
    max_val = max(np.abs(audio).max(), eps)

    return audio / max_val


# Audio playback ###############################################################

def display_audio(data, sr=16000):
    """Listen to the input audio."""
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    return Audio(data=data, rate=sr)


# Plotting functions ###########################################################

def plot_scalogram(scalogram: Tensor,
                   sr: float,
                   y_coords: List[float],
                   title: str = "scalogram",
                   hop_len: int = 1,
                   cmap: str = "magma",
                   vmax: Optional[float] = None,
                   save_path: Optional[str] = None,
                   x_label: str = "Time (seconds)",
                   y_label: str = "Frequency (Hz)") -> None:
    """
    Plots a scalogram of the provided data.

    The scalogram is a visual representation of the wavelet transform of a 
    signal over time. This function uses matplotlib and librosa to create the 
    plot.

    Parameters:
        scalogram (T): The scalogram data to be plotted.
        sr (float): The sample rate of the audio signal.
        y_coords (List[float]): The y-coordinates for the scalogram plot.
        title (str, optional): The title of the plot. Defaults to "scalogram".
        hop_len (int, optional): The hop length for the time axis (or T). 
            Defaults to 1.
        cmap (str, optional): The colormap to use for the plot. Defaults to 
            "magma".
        vmax (Optional[float], optional): The maximum value for the colorbar. 
            If None, the colorbar scales with the data. Defaults to None.
        save_path (Optional[str], optional): The path to save the plot. If 
            None, the plot is not saved. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to 
            "Time (seconds)".
        y_label (str, optional): The label for the y-axis. Defaults to 
            "Frequency (Hz)".
    """
    assert scalogram.ndim == 2
    assert scalogram.size(0) == len(y_coords)
    x_coords = librosa.times_like(scalogram.size(1), sr=sr, hop_length=hop_len)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(scalogram.numpy(),
                             sr=sr,
                             x_axis="time",
                             x_coords=x_coords,
                             y_axis="cqt_hz",
                             y_coords=np.array(y_coords),
                             cmap=cmap,
                             vmin=0.0,
                             vmax=vmax)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if len(y_coords) < 12:
        ax = plt.gca()
        ax.set_yticks(y_coords)
    plt.minorticks_off()
    plt.title(title, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


# Toy signal generation ########################################################

def make_hann_window(n_samples: int) -> Tensor:
    """
    Generate a Hann window of a specified size.

    Parameters:
    -----------
    n_samples : int
        Total number of samples in the Hann window.

    Returns:
    --------
    Tensor
        A tensor containing the generated Hann window.
    """
    x = torch.arange(n_samples)
    y = torch.sin(torch.pi * x / n_samples) ** 2
    return y


def make_pure_sine(n_samples: int, 
                   sr: float, 
                   freq: float, 
                   amp: float = 1.0) -> Tensor:
    """
    Generate a pure sine wave signal of a specified frequency and amplitude.

    Parameters:
    -----------
    n_samples : int
        Total number of samples in the generated signal.
    sr : float
        Sampling rate of the signal.
    freq : float
        Frequency of the sine wave.
    amp : float, optional
        Amplitude of the sine wave. Default is 1.0.

    Returns:
    --------
    Tensor
        A tensor containing the generated sine wave signal.

    Raises:
    -------
    AssertionError
        If the frequency exceeds the Nyquist frequency (sr/2).
    """
    assert freq <= sr / 2.0
    dt = 1.0 / sr
    x = torch.arange(n_samples) * dt
    y = amp * torch.sin(2 * torch.pi * freq * x)
    return y


def make_exp_chirp(n_samples: int,
                   sr: float,
                   start_freq: float = 1.0,
                   end_freq: float = 20000.0,
                   amp: float = 1.0) -> Tensor:
    """
    Generate an exponential chirp signal starting at a specified frequency and 
    ending at another over the duration of the signal.

    Parameters:
    -----------
    n_samples : int
        Total number of samples in the generated signal.
    sr : float
        Sampling rate of the signal.
    start_freq : float, optional
        Start frequency of the chirp. Default is 1.0.
    end_freq : float, optional
        End frequency of the chirp. Default is 20000.0.
    amp : float, optional
        Amplitude of the chirp. Default is 1.0.

    Returns:
    --------
    Tensor
        A tensor containing the generated exponential chirp signal.

    Raises:
    -------
    AssertionError
        If start_freq or end_freq is not within the valid range (1.0 to sr/2).
    """
    assert 1.0 <= start_freq <= sr / 2.0
    assert 1.0 <= end_freq <= sr / 2.0
    if start_freq == end_freq:
        return make_pure_sine(n_samples, sr, start_freq, amp)

    dt = 1.0 / sr
    x = torch.arange(n_samples) * dt
    x_end = x[-1]
    k = x_end / math.log(end_freq / start_freq)
    phase = 2 * torch.pi * k * start_freq * (torch.pow(end_freq / start_freq, x / x_end) - 1.0)
    y = amp * torch.sin(phase)
    return y
