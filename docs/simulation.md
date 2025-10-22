# OSLEC Simulation Environment

## Overview
This workspace provides a Python-based harness for exercising the OSLEC echo
canceller and alternative algorithms without modifying the original kernel
sources. The pipeline builds `echo.c` as a shared library, exposes it through
`ctypes`, and layers scenario generators plus analysis tooling on top.

## Prerequisites
- GCC toolchain for building `liboslec.so`
- Python 3.9+ (recommend a `venv`)
- Python deps: `numpy`, `scipy`, `matplotlib` (see `requirements.txt`)

## Building `liboslec.so`
```bash
cd simulation
make
```
The build uses shim headers under `simulation/include/linux` to satisfy kernel
APIs referenced by `echo.c`.

## Python Environment Setup
```bash
cd simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running a Simulation
```bash
cd simulation
python3 test_suite.py --duration 6 --sample-rate 8000 --noise-snr-db 35
```
Optional flags:
- `--scenario` selects the environment profile (`stationary`,
  `time_variant`, `double_talk`, `tone_interference`)
- `--plot` opens matplotlib visualisations
- `--taps`, `--delay-ms`, `--attenuation-db` tune echo-canceller parameters
- `--verbose` prints debug logging to help diagnose convergence

A typical run prints ERLE statistics and convergence timing. Use `--help` for
all CLI options.

### Choosing Tap Length
Ensure the adaptive filter spans the full echo path:
```
taps >= sample_rate * delay_ms / 1000 + tail_taps
```
Example: `--delay-ms 32` with a 128-tap tail at 8 kHz needs ~384 taps.

## Benchmarking Alternative Algorithms
Compare OSLEC with the reference Python implementations (NLMS, IPNLMS, MDF).
The benchmark automatically sweeps a small hyperparameter grid for each
algorithm (e.g. NLMS: `mu∈{0.6,0.8,1.0}`, IPNLMS: `mu`/`alpha` pairs,
MDF: `block_size`/`mu` combinations) and reports the best-scoring
configuration per scenario:
```bash
cd simulation
python3 benchmark.py --scenario time_variant --taps 384 --duration 3
```
Omit `--scenario` to sweep every profile. Combine with `--verbose` to log
intermediate values. Pass `--plot` to generate comparison plots (microphone,
residual, ERLE) for each scenario. Results list ERLE metrics, residual energy,
and the chosen hyperparameters per algorithm/scenario.

### Scenario Coverage
- `stationary`: baseline echo with Gaussian noise.
- `time_variant`: mid-call delay/attenuation shift.
- `double_talk`: bursty near-end speech superimposed on echo + noise.
- `tone_interference`: injected DTMF/low-frequency tone plus echo.
Use `--scenario` to target individual cases or omit it to exercise them all.

### Benchmark Results (3 s clip, 384 taps)
The table below lists the best-performing hyperparameters (from the default
grid) together with the achieved mean ERLE and residual level for each
scenario.

| Scenario | Algorithm | Mean ERLE (dB) | Residual (dBFS) | Params |
| --- | --- | --- | --- | --- |
| double_talk | OSLEC | 21.03 | −26.28 | – |
|  | NLMS | 3.78 | −24.90 | μ = 0.6 |
|  | IPNLMS | 8.58 | −25.68 | μ = 0.7, α = 0.6 |
|  | MDF | −0.49 | −24.63 | B = 128, μ = 0.4 |
| stationary | OSLEC | 80.71 | −43.65 | – |
|  | NLMS | 25.58 | −46.94 | μ = 0.8 |
|  | IPNLMS | 28.18 | −54.36 | μ = 0.5, α = 0.3 |
|  | MDF | −0.22 | −28.70 | B = 128, μ = 0.4 |
| time_variant | OSLEC | 32.32 | −36.72 | – |
|  | NLMS | 11.22 | −37.12 | μ = 0.8 |
|  | IPNLMS | 14.58 | −39.03 | μ = 0.5, α = 0.3 |
|  | MDF | −0.34 | −32.95 | B = 128, μ = 0.4 |
| tone_interference | OSLEC | 0.52 | −20.17 | – |
|  | NLMS | 0.09 | −18.92 | μ = 0.6 |
|  | IPNLMS | 1.10 | −19.81 | μ = 0.5, α = 0.3 |
|  | MDF | −0.68 | −19.43 | B = 128, μ = 0.4 |

**Interpretation.** OSLEC remains the strongest baseline—especially for
stationary and time-varying echoes—thanks to its dual-path adaptation and
additional processing (NLP/CNG). IPNLMS converges far more quickly (tens of
seconds versus hundreds) and produces lower residual noise in stationary
conditions, but its steady-state ERLE still trails OSLEC. Plain NLMS is a
lightweight fallback once tuned, while the simplified MDF variant requires
further optimisation before it can compete. Tone-interference scenarios remain
challenging for every lightweight algorithm; OSLEC’s additional tone handling
keeps it marginally ahead.

The plotted waveforms/ERLE traces and aggregate bar charts are stored in
`simulation/plots/` (e.g. `double_talk.png`, `benchmark_mean_erle.png`). Regenerate them with:
```bash
cd simulation
source .venv/bin/activate
BENCHMARK_PLOT_DIR=plots MPLBACKEND=Agg BENCHMARK_SAVE=1 \
  python3 benchmark.py --duration 3 --taps 384
```

## Module Layout
- `oslec_wrapper.py` – `ctypes` wrapper with PCM helpers
- `signal_generator.py` – synthetic speech, tones, and noise builders
- `echo_simulator.py` – impulse-response modelling and noise injection
- `scenarios.py` – declarative scenario generation for complex test cases
- `aec_algorithms.py` – Python reference implementations (NLMS, IPNLMS, MDF)
- `benchmark.py` – batch comparison tooling
- `analyzer.py` – ERLE, convergence, and power metrics
- `test_suite.py` – interactive runner for OSLEC under a single scenario

## IPNLMS Drop-in Replacement
Build the alternative shared library and point the simulation toolkit (or your
own application) to it via `OSLEC_LIB`:
```bash
cd simulation
make ipnlms
OSLEC_LIB=$(pwd)/liboslec_ipnlms.so python3 test_suite.py --scenario time_variant --taps 384
```
The IPNLMS drop-in converges faster but currently trails stock OSLEC in steady
state ERLE (≈16–17 dB vs ≈30 dB in the `time_variant` scenario). Tune the
constants inside `echo_ipnlms.c` (`DEFAULT_MU_Q16`, `DEFAULT_ALPHA_Q16`) to
balance speed and suppression for your hardware.

## Next Steps
- Add perceptual quality metrics (PESQ/POLQA)
- Extend benchmarking to produce CSV summaries
- Integrate automated sweeps for regression tracking
