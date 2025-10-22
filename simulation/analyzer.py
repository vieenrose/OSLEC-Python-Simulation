"""
Analysis helpers to evaluate OSLEC performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def windowed_power(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(signal) < window_size:
        padded = np.pad(signal, (0, window_size - len(signal)), mode="constant")
        return np.array([np.mean(padded**2)])
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(signal**2, kernel, mode="valid")


def erle_db(
    microphone_signal: np.ndarray,
    residual_signal: np.ndarray,
    window_size: int = 1024,
) -> np.ndarray:
    mic_power = windowed_power(microphone_signal, window_size)
    res_power = windowed_power(residual_signal, window_size)
    return 10.0 * np.log10((mic_power + EPS) / (res_power + EPS))


def mean_erle_db(erle_trace: np.ndarray) -> float:
    return float(np.mean(erle_trace))


def convergence_time_seconds(
    erle_trace: np.ndarray,
    sample_rate: int,
    window_size: int,
    threshold_db: float,
    required_windows: int = 5,
) -> float | None:
    if len(erle_trace) < required_windows:
        return None

    for index in range(len(erle_trace) - required_windows + 1):
        window = erle_trace[index : index + required_windows]
        if np.all(window >= threshold_db):
            hop = 1.0 / sample_rate
            samples_per_window = window_size
            time_seconds = (index * hop * samples_per_window) + (
                window_size / sample_rate
            )
            return float(time_seconds)
    return None


def residual_echo_db(residual_signal: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(residual_signal**2) + EPS)


def far_end_power_db(far_signal: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(far_signal**2) + EPS)


@dataclass
class SimulationMetrics:
    mean_erle: float
    peak_erle: float
    convergence_time: float | None
    residual_echo: float
    far_end_power: float


def summarize_metrics(
    microphone_signal: np.ndarray,
    residual_signal: np.ndarray,
    far_reference: np.ndarray,
    sample_rate: int,
    window_size: int = 1024,
    threshold_db: float = 20.0,
) -> SimulationMetrics:
    erle_trace = erle_db(microphone_signal, residual_signal, window_size)
    mean_erle = mean_erle_db(erle_trace)
    peak_erle = float(np.max(erle_trace))
    convergence = convergence_time_seconds(
        erle_trace, sample_rate, window_size, threshold_db
    )
    residual = residual_echo_db(residual_signal)
    far_power = far_end_power_db(far_reference)
    return SimulationMetrics(mean_erle, peak_erle, convergence, residual, far_power)
