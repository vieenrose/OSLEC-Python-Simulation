"""High level scenarios to exercise the OSLEC simulation stack."""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from analyzer import SimulationMetrics, erle_db, summarize_metrics
from oslec_wrapper import (
    AdaptionMode,
    OSLEC,
    float_to_pcm16,
    pcm16_to_float,
)
from scenarios import build_signals, list_scenarios, required_taps


LOG = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    sample_rate: int = 8000
    duration_s: float = 6.0
    taps: int = 128
    adaption_mode: AdaptionMode = AdaptionMode.default()
    scenario: str = "stationary"
    echo_delay_ms: float = 32.0
    echo_attenuation_db: float = 12.0
    echo_decay: float = 0.45
    tail_taps: int = 128
    variant_delay_ms: float = 48.0
    variant_attenuation_db: float = 16.0
    variant_decay: float = 0.35
    near_level: float = 0.7
    far_gain: float = 1.0
    near_gain: float = 1.0
    tone_digit: str = "5"
    tone_level: float = 0.4
    noise_snr_db: float = 35.0
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


def run_simulation(config: SimulationConfig) -> SimulationResult:
    LOG.debug(
        "Starting simulation: scenario=%s duration=%.2fs sr=%d taps=%d",
        config.scenario,
        config.duration_s,
        config.sample_rate,
        config.taps,
    )

    scenario_config = asdict(config)
    tx_signal, microphone, metadata = build_signals(
        config.scenario, {**scenario_config}
    )
    LOG.debug("Scenario metadata: %s", metadata)

    needed = required_taps(
        config.sample_rate, config.echo_delay_ms, config.tail_taps
    )
    if config.taps < needed:
        LOG.warning(
            "taps (%d) shorter than echo path (%d samples ≈ %.2f ms)."
            " Increase --taps or reduce --delay-ms/--tail-taps",
            config.taps,
            needed,
            needed / config.sample_rate * 1000.0,
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
    scenario_names = sorted(list_scenarios().keys())
    parser.add_argument("--scenario", choices=scenario_names, default="stationary")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--taps", type=int, default=128)
    parser.add_argument("--delay-ms", type=float, default=32.0)
    parser.add_argument("--attenuation-db", type=float, default=12.0)
    parser.add_argument("--decay", type=float, default=0.45)
    parser.add_argument("--variant-delay-ms", type=float, default=48.0)
    parser.add_argument("--variant-attenuation-db", type=float, default=16.0)
    parser.add_argument("--variant-decay", type=float, default=0.35)
    parser.add_argument("--noise-snr-db", type=float, default=35.0)
    parser.add_argument("--near-level", type=float, default=0.7)
    parser.add_argument("--far-gain", type=float, default=1.0)
    parser.add_argument("--near-gain", type=float, default=1.0)
    parser.add_argument("--tone-digit", type=str, default="5")
    parser.add_argument("--tone-level", type=float, default=0.4)
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
        scenario=args.scenario,
        duration_s=args.duration,
        sample_rate=args.sample_rate,
        taps=args.taps,
        echo_delay_ms=args.delay_ms,
        echo_attenuation_db=args.attenuation_db,
        echo_decay=args.decay,
        variant_delay_ms=args.variant_delay_ms,
        variant_attenuation_db=args.variant_attenuation_db,
        variant_decay=args.variant_decay,
        near_level=args.near_level,
        far_gain=args.far_gain,
        near_gain=args.near_gain,
        tone_digit=args.tone_digit,
        tone_level=args.tone_level,
        noise_snr_db=args.noise_snr_db,
        seed=args.seed,
    )
    result = run_simulation(config)
    print_report(result)
    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
