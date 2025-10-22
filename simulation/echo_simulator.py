"""
Utilities for injecting echo paths and synthesising microphone signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


LOG = logging.getLogger(__name__)


@dataclass
class EchoPath:
    delay_ms: float
    attenuation_db: float
    sample_rate: int
    decay: float = 0.4
    tail_taps: int = 128

    def impulse_response(self) -> np.ndarray:
        delay_samples = int(round(self.delay_ms * self.sample_rate / 1000.0))
        total_length = delay_samples + self.tail_taps
        impulse = np.zeros(total_length, dtype=np.float32)
        base_gain = 10.0 ** (-self.attenuation_db / 20.0)
        impulse[delay_samples] = base_gain

        for tap in range(1, self.tail_taps):
            impulse[delay_samples + tap] = base_gain * (self.decay ** tap)
        LOG.debug(
            "EchoPath impulse: delay_samples=%d tail_taps=%d attenuation=%.1f "
            "decay=%.2f",
            delay_samples,
            self.tail_taps,
            self.attenuation_db,
            self.decay,
        )
        return impulse


def simulate_echo(
    far_end_signal: np.ndarray, echo_path: EchoPath
) -> Tuple[np.ndarray, np.ndarray]:
    impulse = echo_path.impulse_response()
    echo = signal.fftconvolve(far_end_signal, impulse)[: len(far_end_signal)]
    microphone = echo.copy()
    LOG.debug(
        "Simulated echo: samples=%d echo_peak=%.4f",
        len(echo),
        float(np.max(np.abs(echo)) if len(echo) else 0.0),
    )
    return echo, microphone


def add_background_noise(
    signal_in: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if snr_db == np.inf:
        return signal_in

    generator = rng or np.random.default_rng()
    noise = generator.normal(0.0, 1.0, size=signal_in.shape)
    signal_power = np.mean(signal_in**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    desired_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(desired_noise_power / noise_power)
    combined = signal_in + scaled_noise
    LOG.debug(
        "Added background noise: snr_db=%.1f signal_power=%.4e noise_power=%.4e",
        snr_db,
        signal_power,
        desired_noise_power,
    )
    return combined


def prepare_double_talk(
    far_end_ref: np.ndarray,
    near_end_signal: np.ndarray,
    far_gain: float,
    near_gain: float,
) -> Tuple[np.ndarray, np.ndarray]:
    length = max(len(far_end_ref), len(near_end_signal))
    far = np.pad(far_end_ref, (0, length - len(far_end_ref))) * far_gain
    near = np.pad(near_end_signal, (0, length - len(near_end_signal))) * near_gain
    microphone = far + near
    LOG.debug(
        "Prepared double-talk mix: length=%d far_gain=%.2f near_gain=%.2f",
        length,
        far_gain,
        near_gain,
    )
    return far, microphone
