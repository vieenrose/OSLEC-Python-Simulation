"""
Scenario generation utilities for OSLEC simulations.

Each scenario returns transmit (tx) and microphone (mic) signals along with
metadata describing the configuration used to produce them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from echo_simulator import EchoPath, add_background_noise, prepare_double_talk
from signal_generator import (
    DoubleTalkConfig,
    band_limited_noise,
    dtmf_tone,
    speech_like_signal,
    white_noise,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    generator: Callable[[Dict], Tuple[np.ndarray, np.ndarray, Dict]]


def _memoized_rng(config: Dict) -> np.random.Generator:
    seed = config.get("seed")
    return np.random.default_rng(seed)


def _base_echo_path(config: Dict) -> EchoPath:
    return EchoPath(
        delay_ms=config.get("echo_delay_ms", 32.0),
        attenuation_db=config.get("echo_attenuation_db", 12.0),
        sample_rate=config["sample_rate"],
        decay=config.get("echo_decay", 0.45),
        tail_taps=config.get("tail_taps", 128),
    )


def stationary_echo(config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rng = _memoized_rng(config)
    tx = speech_like_signal(
        config["duration_s"],
        config["sample_rate"],
        amplitude=config.get("tx_level", 0.95),
        rng=rng,
    )
    path = _base_echo_path(config)
    impulse = path.impulse_response()
    echo = np.convolve(tx, impulse, mode="full")[: len(tx)]
    microphone = add_background_noise(
        echo,
        config.get("noise_snr_db", 35.0),
        rng=rng,
    )
    metadata = {
        "echo_impulse": impulse,
        "rng_seed": config.get("seed"),
        "description": "Stationary echo path with steady far-end speech.",
    }
    return tx, microphone, metadata


def time_variant_echo(config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rng = _memoized_rng(config)
    duration = config["duration_s"]
    sample_rate = config["sample_rate"]
    tx = speech_like_signal(
        duration,
        sample_rate,
        amplitude=config.get("tx_level", 0.95),
        rng=rng,
    )
    half = len(tx) // 2

    path_a = _base_echo_path(config)
    path_b = EchoPath(
        delay_ms=config.get("variant_delay_ms", 48.0),
        attenuation_db=config.get("variant_attenuation_db", 16.0),
        sample_rate=sample_rate,
        decay=config.get("variant_decay", 0.35),
        tail_taps=config.get("tail_taps", 128),
    )

    impulse_a = path_a.impulse_response()
    impulse_b = path_b.impulse_response()

    echo_a = np.convolve(tx[:half], impulse_a, mode="full")[:half]
    echo_b = np.convolve(tx[half:], impulse_b, mode="full")[: len(tx) - half]
    echo = np.concatenate([echo_a, echo_b])

    microphone = add_background_noise(
        echo,
        config.get("noise_snr_db", 30.0),
        rng=rng,
    )

    metadata = {
        "impulse_first": impulse_a,
        "impulse_second": impulse_b,
        "transition_sample": half,
        "description": "Echo path changes halfway through the call.",
    }
    return tx, microphone, metadata


def double_talk_bursts(config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rng = _memoized_rng(config)
    sample_rate = config["sample_rate"]
    duration = config["duration_s"]

    far_tx = speech_like_signal(
        duration,
        sample_rate,
        amplitude=config.get("tx_level", 0.9),
        rng=rng,
    )
    near_signal = speech_like_signal(
        duration,
        sample_rate,
        amplitude=config.get("near_level", 0.7),
        rng=rng,
    )

    # Gate near-end activity to create bursts.
    gating = np.zeros_like(near_signal)
    burst_len = int(0.25 * sample_rate)
    interval = int(0.6 * sample_rate)
    pos = int(0.4 * sample_rate)
    while pos < len(gating):
        gating[pos : pos + burst_len] = 1.0
        pos += interval
    near_signal *= gating

    path = _base_echo_path(config)
    impulse = path.impulse_response()
    far_echo = np.convolve(far_tx, impulse, mode="full")[: len(far_tx)]
    _, microphone = prepare_double_talk(
        far_echo,
        near_signal,
        far_gain=config.get("far_gain", 1.0),
        near_gain=config.get("near_gain", 1.0),
    )

    microphone = add_background_noise(
        microphone,
        config.get("noise_snr_db", 25.0),
        rng=rng,
    )

    metadata = {
        "echo_impulse": impulse,
        "near_burst_mask": gating,
        "description": "Double-talk bursts with background noise.",
    }
    return far_tx, microphone, metadata


def tone_interference(config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rng = _memoized_rng(config)
    sample_rate = config["sample_rate"]
    duration = config["duration_s"]
    tx = speech_like_signal(
        duration,
        sample_rate,
        amplitude=config.get("tx_level", 0.8),
        rng=rng,
    )

    path = _base_echo_path(config)
    impulse = path.impulse_response()
    echo = np.convolve(tx, impulse, mode="full")[: len(tx)]

    tone_duration = min(1.0, duration / 3.0)
    tone = dtmf_tone(config.get("tone_digit", "5"), tone_duration, sample_rate)
    tone_pad = np.pad(
        tone,
        (
            int(sample_rate * 0.5),
            len(tx) - len(tone) - int(sample_rate * 0.5),
        ),
    )
    microphone = echo + config.get("tone_level", 0.4) * tone_pad

    noise = band_limited_noise(
        duration,
        sample_rate,
        low_cut_hz=100.0,
        high_cut_hz=700.0,
        rms=0.03,
        rng=rng,
    )
    microphone += noise

    metadata = {
        "echo_impulse": impulse,
        "tone_digit": config.get("tone_digit", "5"),
        "description": "Echo mixed with low-frequency tone interference.",
    }
    return tx, microphone, metadata


SCENARIOS: Dict[str, Scenario] = {
    "stationary": Scenario(
        name="stationary",
        description="Stationary echo path with Gaussian noise.",
        generator=stationary_echo,
    ),
    "time_variant": Scenario(
        name="time_variant",
        description="Echo path delay/attenuation change mid-call.",
        generator=time_variant_echo,
    ),
    "double_talk": Scenario(
        name="double_talk",
        description="Intermittent near-end bursts with background noise.",
        generator=double_talk_bursts,
    ),
    "tone_interference": Scenario(
        name="tone_interference",
        description="Echo plus interfering tone and low-frequency noise.",
        generator=tone_interference,
    ),
}


def list_scenarios() -> Dict[str, Scenario]:
    return SCENARIOS


def build_signals(scenario_key: str, config: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    scenario = SCENARIOS.get(scenario_key)
    if scenario is None:
        raise ValueError(f"Unknown scenario '{scenario_key}'")
    tx, microphone, meta = scenario.generator(config)
    return tx, microphone, meta
