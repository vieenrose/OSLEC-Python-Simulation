"""High level scenarios to exercise the OSLEC simulation stack."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from analyzer import SimulationMetrics, erle_db, summarize_metrics
from echo_simulator import EchoPath, add_background_noise, prepare_double_talk
from oslec_wrapper import (
    AdaptionMode,
    OSLEC,
    float_to_pcm16,
    pcm16_to_float,
)
from signal_generator import DoubleTalkConfig, speech_like_signal


LOG = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    sample_rate: int = 8000
    duration_s: float = 6.0
    taps: int = 128
    adaption_mode: AdaptionMode = AdaptionMode.default()
    echo_delay_ms: float = 32.0
    echo_attenuation_db: float = 12.0
    echo_decay: float = 0.45
    noise_snr_db: float = 35.0
    include_double_talk: bool = False
    double_talk: DoubleTalkConfig = field(default_factory=DoubleTalkConfig)
    seed: Optional[int] = None
    window_size: int = 1024
    erle_threshold_db: float = 20.0


@dataclass
class SimulationResult:
    config: SimulationConfig
    tx_signal: np.ndarray
    microphone_signal: np.ndarray
    residual_signal: np.ndarray
    metrics: SimulationMetrics
    erle_trace: np.ndarray


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if verbose:
        LOG.debug("Verbose logging enabled")


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def run_simulation(config: SimulationConfig) -> SimulationResult:
    rng = _rng(config.seed)

    LOG.debug(
        "Starting simulation: duration=%.2fs sr=%d taps=%d delay=%.1fms "
        "atten=%.1fdB decay=%.2f noise_snr=%.1fdB double_talk=%s",
        config.duration_s,
        config.sample_rate,
        config.taps,
        config.echo_delay_ms,
        config.echo_attenuation_db,
        config.echo_decay,
        config.noise_snr_db,
        config.include_double_talk,
    )

    tx_signal = speech_like_signal(
        config.duration_s,
        config.sample_rate,
        amplitude=0.95,
        rng=rng,
    )
    echo_path = EchoPath(
        delay_ms=config.echo_delay_ms,
        attenuation_db=config.echo_attenuation_db,
        sample_rate=config.sample_rate,
        decay=config.echo_decay,
    )
    LOG.debug(
        "Generated echo path (delay_samples=%d, tail_taps=%d)",
        int(round(config.echo_delay_ms * config.sample_rate / 1000.0)),
        echo_path.tail_taps,
    )
    echo_signal = np.convolve(
        tx_signal,
        echo_path.impulse_response(),
        mode="full",
    )[: len(tx_signal)]
    microphone = echo_signal.copy()

    if config.include_double_talk:
        near_end = speech_like_signal(
            config.duration_s, config.sample_rate, amplitude=0.6, rng=rng
        )
        _, microphone = prepare_double_talk(
            echo_signal,
            near_end,
            far_gain=config.double_talk.far_end_level,
            near_gain=config.double_talk.near_end_level,
        )
        LOG.debug(
            "Applied double-talk: near_end_level=%.2f far_end_level=%.2f",
            config.double_talk.near_end_level,
            config.double_talk.far_end_level,
        )

    microphone = add_background_noise(
        microphone, config.noise_snr_db, rng=rng
    )
    LOG.debug(
        "Added background noise: post_power=%.4e",
        float(np.mean(microphone**2)),
    )
    tx_signal = np.clip(tx_signal, -1.0, 0.999969482421875)
    microphone = np.clip(microphone, -1.0, 0.999969482421875)

    if len(microphone) > len(tx_signal):
        tx_signal = np.pad(
            tx_signal, (0, len(microphone) - len(tx_signal)), mode="constant"
        )
    elif len(microphone) < len(tx_signal):
        microphone = np.pad(
            microphone,
            (0, len(tx_signal) - len(microphone)),
            mode="constant",
        )

    tx_pcm = float_to_pcm16(tx_signal)
    mic_pcm = float_to_pcm16(microphone)

    with OSLEC(config.taps, config.adaption_mode) as canceller:
        residual_pcm = canceller.process(tx_pcm, mic_pcm)
        LOG.debug(
            "Finished OSLEC processing: frames=%d adaption_mode=0x%02x",
            len(residual_pcm),
            int(config.adaption_mode),
        )

    residual = pcm16_to_float(residual_pcm)
    metrics = summarize_metrics(
        microphone,
        residual,
        tx_signal,
        sample_rate=config.sample_rate,
        window_size=config.window_size,
        threshold_db=config.erle_threshold_db,
    )
    erle_trace = erle_db(
        microphone, residual, window_size=config.window_size
    )
    LOG.debug(
        "Computed metrics preview: mean_erle=%.2f peak_erle=%.2f "
        "residual_db=%.2f",
        metrics.mean_erle,
        metrics.peak_erle,
        metrics.residual_echo,
    )

    return SimulationResult(
        config=config,
        tx_signal=tx_signal,
        microphone_signal=microphone,
        residual_signal=residual,
        metrics=metrics,
        erle_trace=erle_trace,
    )


def print_report(result: SimulationResult) -> None:
    metrics = result.metrics
    print("Simulation metrics:")
    print(f"  Mean ERLE:       {metrics.mean_erle:6.2f} dB")
    print(f"  Peak ERLE:       {metrics.peak_erle:6.2f} dB")
    LOG.debug(
        "Metrics detail: mean_erle=%.2f peak_erle=%.2f residual=%.2f "
        "far_power=%.2f convergence=%s",
        metrics.mean_erle,
        metrics.peak_erle,
        metrics.residual_echo,
        metrics.far_end_power,
        metrics.convergence_time,
    )
    convergence = (
        f"{metrics.convergence_time:6.3f} s"
        if metrics.convergence_time is not None
        else "   N/A"
    )
    print(f"  Convergence:     {convergence}")
    print(f"  Residual Echo:   {metrics.residual_echo:6.2f} dBFS")
    print(f"  Far-end Power:   {metrics.far_end_power:6.2f} dBFS")


def plot_result(result: SimulationResult) -> None:
    import matplotlib.pyplot as plt

    config = result.config
    duration = len(result.microphone_signal) / config.sample_rate
    time_axis = np.linspace(0.0, duration, len(result.microphone_signal))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(time_axis, result.tx_signal, label="Tx")
    axes[0].set_ylabel("Tx")
    axes[0].legend()

    axes[1].plot(time_axis, result.microphone_signal, label="Microphone")
    axes[1].set_ylabel("Mic")
    axes[1].legend()

    axes[2].plot(time_axis, result.residual_signal, label="Residual")
    axes[2].set_ylabel("Residual")
    axes[2].legend()

    erle_time = np.linspace(
        0.0,
        duration,
        len(result.erle_trace),
    )
    axes[3].plot(erle_time, result.erle_trace, label="ERLE (dB)")
    axes[3].axhline(
        config.erle_threshold_db, color="red", linestyle="--", label="Target"
    )
    axes[3].set_ylabel("ERLE (dB)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()

    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSLEC echo cancellation simulations."
    )
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--taps", type=int, default=128)
    parser.add_argument("--delay-ms", type=float, default=32.0)
    parser.add_argument("--attenuation-db", type=float, default=12.0)
    parser.add_argument("--decay", type=float, default=0.45)
    parser.add_argument("--noise-snr-db", type=float, default=35.0)
    parser.add_argument("--double-talk", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable debug logging for troubleshooting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)
    config = SimulationConfig(
        duration_s=args.duration,
        sample_rate=args.sample_rate,
        taps=args.taps,
        echo_delay_ms=args.delay_ms,
        echo_attenuation_db=args.attenuation_db,
        echo_decay=args.decay,
        noise_snr_db=args.noise_snr_db,
        include_double_talk=args.double_talk,
        seed=args.seed,
    )
    result = run_simulation(config)
    print_report(result)
    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
