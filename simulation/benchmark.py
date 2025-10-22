"""
Benchmark multiple AEC algorithms under various scenarios.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from aec_algorithms import (
    IPNLMSAEC,
    IPNLMSParams,
    MDFAEC,
    MDFParams,
    NLMSAEC,
    NLMSParams,
    OSLECWrapper,
)
from analyzer import erle_db, summarize_metrics
from scenarios import build_signals, list_scenarios


LOG = logging.getLogger(__name__)


@dataclass
class AlgorithmSpec:
    key: str
    label: str
    builder: Callable[..., object]
    param_grid: List[Dict]


def build_algorithms(taps: int) -> Dict[str, AlgorithmSpec]:
    return {
        "oslec": AlgorithmSpec(
            key="oslec",
            label="OSLEC",
            builder=lambda **kwargs: OSLECWrapper(taps),
            param_grid=[{}],
        ),
        "nlms": AlgorithmSpec(
            key="nlms",
            label="NLMS",
            builder=lambda **params: NLMSAEC(
                NLMSParams(
                    taps=taps,
                    mu=params.get("mu", 0.7),
                    epsilon=params.get("epsilon", 1e-6),
                )
            ),
            param_grid=[
                {"mu": 0.6},
                {"mu": 0.8},
                {"mu": 1.0},
            ],
        ),
        "ipnlms": AlgorithmSpec(
            key="ipnlms",
            label="IPNLMS",
            builder=lambda **params: IPNLMSAEC(
                IPNLMSParams(
                    taps=taps,
                    mu=params.get("mu", 0.6),
                    alpha=params.get("alpha", 0.5),
                    epsilon=params.get("epsilon", 1e-6),
                )
            ),
            param_grid=[
                {"mu": 0.5, "alpha": 0.3},
                {"mu": 0.6, "alpha": 0.5},
                {"mu": 0.7, "alpha": 0.6},
            ],
        ),
        "mdf": AlgorithmSpec(
            key="mdf",
            label="MDF",
            builder=lambda **params: MDFAEC(
                MDFParams(
                    taps=taps,
                    block_size=params.get("block_size", 128),
                    mu=params.get("mu", 0.5),
                    epsilon=params.get("epsilon", 1e-6),
                )
            ),
            param_grid=[
                {"block_size": 128, "mu": 0.4},
                {"block_size": 128, "mu": 0.5},
                {"block_size": 256, "mu": 0.35},
            ],
        ),
    }


def plot_results(
    scenario_key: str,
    sample_rate: int,
    window_size: int,
    entries: List[Dict],
    tx: np.ndarray,
    mic: np.ndarray,
    threshold_db: float,
    output_dir: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    duration = len(mic) / sample_rate
    time_axis = np.linspace(0.0, duration, len(mic))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    title_lines = [f"Scenario: {scenario_key}"]
    for entry in entries:
        params = entry.get("params") or {}
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            title_lines.append(f"{entry['algorithm']}: {param_str}")
    fig.suptitle(" | ".join(title_lines))

    axes[0].plot(time_axis, mic, label="Microphone")
    axes[0].set_ylabel("Mic")
    axes[0].legend()

    for entry in entries:
        axes[1].plot(time_axis, entry["residual"], label=entry["algorithm"])
    axes[1].set_ylabel("Residual")
    axes[1].legend()

    axes[2].axhline(threshold_db, color="red", linestyle="--", label="Target")
    for entry in entries:
        erle = entry["erle_trace"]
        if len(erle) == 0:
            continue
        erle_time = (
            (np.arange(len(erle)) + window_size / 2.0) / sample_rate
        )
        axes[2].plot(erle_time, erle, label=entry["algorithm"])
    axes[2].set_ylabel("ERLE (dB)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{scenario_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def run_benchmark(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    scenarios = list_scenarios()
    selected_scenarios = (
        [args.scenario] if args.scenario else sorted(scenarios.keys())
    )
    algorithms = build_algorithms(args.taps)

    results: List[Dict] = []
    scenario_data: Dict[str, Dict] = {}

    for scenario_key in selected_scenarios:
        scenario = scenarios[scenario_key]
        LOG.info("Scenario: %s - %s", scenario.name, scenario.description)
        tx, mic, meta = build_signals(
            scenario_key,
            {
                "sample_rate": args.sample_rate,
                "duration_s": args.duration,
                "echo_delay_ms": args.delay_ms,
                "echo_attenuation_db": args.attenuation_db,
                "echo_decay": args.decay,
                "variant_delay_ms": args.variant_delay_ms,
                "variant_attenuation_db": args.variant_attenuation_db,
                "variant_decay": args.variant_decay,
                "tail_taps": args.tail_taps,
                "noise_snr_db": args.noise_snr_db,
                "seed": args.seed,
                "near_level": args.near_level,
                "far_gain": args.far_gain,
                "near_gain": args.near_gain,
                "tone_digit": args.tone_digit,
                "tone_level": args.tone_level,
            },
        )

        scenario_entries: List[Dict] = []

        for alg_key, spec in algorithms.items():
            if args.algorithms and alg_key not in args.algorithms:
                continue
            best_entry = None

            for params in spec.param_grid:
                aec = spec.builder(**params)
                LOG.info("  Running %s params=%s", spec.label, params)
                residual = aec.process(tx, mic)
                metrics = summarize_metrics(
                    mic,
                    residual,
                    tx,
                    sample_rate=args.sample_rate,
                    window_size=args.window_size,
                    threshold_db=args.threshold_db,
                )
                erle_trace = erle_db(
                    mic,
                    residual,
                    window_size=args.window_size,
                )
                entry = {
                    "algorithm": spec.label,
                    "residual": residual,
                    "metrics": metrics,
                    "erle_trace": erle_trace,
                    "params": dict(params),
                }
                if (
                    best_entry is None
                    or metrics.mean_erle > best_entry["metrics"].mean_erle
                ):
                    best_entry = entry

            if best_entry is None:
                continue

            scenario_entries.append(best_entry)
            metrics = best_entry["metrics"]
            results.append(
                {
                    "scenario": scenario_key,
                    "algorithm": spec.label,
                    "mean_erle": metrics.mean_erle,
                    "peak_erle": metrics.peak_erle,
                    "convergence": metrics.convergence_time,
                    "residual_db": metrics.residual_echo,
                    "params": best_entry["params"],
                }
            )

        scenario_data[scenario_key] = {
            "tx": tx,
            "mic": mic,
            "entries": scenario_entries,
        }

    print("\n=== Benchmark Summary ===")
    for entry in results:
        convergence = (
            f"{entry['convergence']:.3f}s"
            if entry["convergence"] is not None
            else "N/A"
        )
        params = entry.get("params") or {}
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            params_fragment = f" params [{params_str}]"
        else:
            params_fragment = ""
        print(
            f"{entry['scenario']:>14} | {entry['algorithm']:>6} | "
            f"mean ERLE {entry['mean_erle']:6.2f} dB | "
            f"peak {entry['peak_erle']:6.2f} dB | "
            f"conv {convergence} | residual {entry['residual_db']:7.2f} dBFS"
            f"{params_fragment}"
        )

    output_dir = os.environ.get("BENCHMARK_PLOT_DIR")

    if args.plot or output_dir:
        for scenario_key in selected_scenarios:
            data = scenario_data.get(scenario_key)
            if not data or not data["entries"]:
                continue
            plot_results(
                scenario_key,
                args.sample_rate,
                args.window_size,
                data["entries"],
                data["tx"],
                data["mic"],
                args.threshold_db,
                output_dir=output_dir,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple AEC algorithms."
    )
    parser.add_argument("--scenario", choices=list_scenarios().keys())
    parser.add_argument("--algorithms", nargs="*", choices=list(build_algorithms(1).keys()))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--taps", type=int, default=384)
    parser.add_argument("--delay-ms", type=float, default=32.0)
    parser.add_argument("--attenuation-db", type=float, default=12.0)
    parser.add_argument("--decay", type=float, default=0.45)
    parser.add_argument("--variant-delay-ms", type=float, default=48.0)
    parser.add_argument("--variant-attenuation-db", type=float, default=16.0)
    parser.add_argument("--variant-decay", type=float, default=0.35)
    parser.add_argument("--tail-taps", type=int, default=128)
    parser.add_argument("--noise-snr-db", type=float, default=30.0)
    parser.add_argument("--near-level", type=float, default=0.7)
    parser.add_argument("--far-gain", type=float, default=1.0)
    parser.add_argument("--near-gain", type=float, default=1.0)
    parser.add_argument("--tone-digit", type=str, default="5")
    parser.add_argument("--tone-level", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--threshold-db", type=float, default=25.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
