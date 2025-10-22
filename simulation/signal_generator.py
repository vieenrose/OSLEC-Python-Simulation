"""
Signal generation helpers for the OSLEC simulation environment.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


LOG = logging.getLogger(__name__)


def generate_time_axis(duration_s: float, sample_rate: int) -> np.ndarray:
    total_samples = int(round(duration_s * sample_rate))
    return np.arange(total_samples, dtype=np.float64) / float(sample_rate)


def sine_wave(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.9,
    phase: float = 0.0,
) -> np.ndarray:
    time_axis = generate_time_axis(duration_s, sample_rate)
    return amplitude * np.sin(2.0 * math.pi * frequency_hz * time_axis + phase)


def white_noise(
    duration_s: float,
    sample_rate: int,
    rms: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    generator = rng or np.random.default_rng()
    samples = generator.normal(0.0, 1.0, int(round(duration_s * sample_rate)))
    power = np.sqrt(np.mean(samples**2) + 1e-12)
    LOG.debug(
        "Generated white noise: duration=%.2fs sr=%d rms=%.3f",
        duration_s,
        sample_rate,
        rms,
    )
    return (samples / power) * rms


def band_limited_noise(
    duration_s: float,
    sample_rate: int,
    low_cut_hz: float,
    high_cut_hz: float,
    rms: float = 0.03,
    order: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    noise = white_noise(duration_s, sample_rate, rms=1.0, rng=rng)
    sos = signal.butter(
        order,
        [low_cut_hz, high_cut_hz],
        btype="band",
        output="sos",
        fs=sample_rate,
    )
    filtered = signal.sosfilt(sos, noise)
    power = np.sqrt(np.mean(filtered**2) + 1e-12)
    LOG.debug(
        "Generated band-limited noise: %.1f-%.1f Hz order=%d rms=%.3f",
        low_cut_hz,
        high_cut_hz,
        order,
        rms,
    )
    return (filtered / power) * rms


def speech_like_signal(
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    noise = band_limited_noise(
        duration_s,
        sample_rate,
        low_cut_hz=200.0,
        high_cut_hz=3400.0,
        rms=1.0,
        rng=rng,
    )
    envelope = np.abs(signal.hilbert(noise))
    envelope /= np.max(envelope + 1e-9)
    shaped = noise * envelope
    shaped /= np.max(np.abs(shaped) + 1e-9)
    LOG.debug(
        "Generated speech-like signal: duration=%.2fs sr=%d amplitude=%.2f",
        duration_s,
        sample_rate,
        amplitude,
    )
    return shaped * amplitude


def dtmf_tone(
    digit: str,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.8,
) -> np.ndarray:
    row_freqs = {
        "1": 697,
        "2": 697,
        "3": 697,
        "A": 697,
        "4": 770,
        "5": 770,
        "6": 770,
        "B": 770,
        "7": 852,
        "8": 852,
        "9": 852,
        "C": 852,
        "*": 941,
        "0": 941,
        "#": 941,
        "D": 941,
    }
    col_freqs = {
        "1": 1209,
        "2": 1336,
        "3": 1477,
        "A": 1633,
        "4": 1209,
        "5": 1336,
        "6": 1477,
        "B": 1633,
        "7": 1209,
        "8": 1336,
        "9": 1477,
        "C": 1633,
        "*": 1209,
        "0": 1336,
        "#": 1477,
        "D": 1633,
    }
    if digit not in row_freqs:
        raise ValueError(f"Unsupported DTMF digit '{digit}'")

    t = generate_time_axis(duration_s, sample_rate)
    waveform = np.sin(2.0 * math.pi * row_freqs[digit] * t)
    waveform += np.sin(2.0 * math.pi * col_freqs[digit] * t)
    waveform /= np.max(np.abs(waveform) + 1e-9)
    LOG.debug(
        "Generated DTMF tone: digit=%s duration=%.2fs sr=%d amplitude=%.2f",
        digit,
        duration_s,
        sample_rate,
        amplitude,
    )
    return amplitude * waveform


@dataclass(frozen=True)
class DoubleTalkConfig:
    near_end_delay_ms: float = 0.0
    near_end_level: float = 1.0
    far_end_level: float = 1.0


def combine_double_talk(
    far_end: np.ndarray,
    near_end: np.ndarray,
    sample_rate: int,
    config: DoubleTalkConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mix two signals into near-end microphone input and transmit reference.
    """
    delay_samples = int(round(config.near_end_delay_ms * sample_rate / 1000.0))
    far = far_end * config.far_end_level
    near = near_end * config.near_end_level

    if delay_samples > 0:
        near = np.concatenate(
            [np.zeros(delay_samples, dtype=near.dtype), near]
        )

    LOG.debug(
        "Combining double talk: delay_samples=%d far_level=%.2f near_level=%.2f",
        delay_samples,
        config.far_end_level,
        config.near_end_level,
    )
    max_len = max(len(far), len(near))
    far = np.pad(far, (0, max_len - len(far)))
    near = np.pad(near, (0, max_len - len(near)))
    return far, near
