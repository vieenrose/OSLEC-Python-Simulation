# OSLEC Simulation Environment

## Overview
A user-space harness for exercising the OSLEC echo canceller without touching
the original kernel-oriented sources. The workflow builds `echo.c` as a shared
library, exposes it through `ctypes`, and ships helper modules for signal
generation, echo-path modelling, and performance analysis.

## Prerequisites
- GCC toolchain for building the shared library
- Python 3.9+ with `venv`
- `numpy`, `scipy`, and `matplotlib` (install via the provided requirements)

## Building `liboslec.so`
```bash
cd simulation
make     # produces liboslec.so next to the Python sources
```
The build relies on lightweight stub headers in `simulation/include/linux` to
satisfy the kernel dependencies used by `echo.c`.

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
- `--double-talk` enables simultaneous near-end speech
- `--plot` opens matplotlib visualisations
- `--taps`, `--delay-ms`, `--attenuation-db` tune echo-canceller parameters
- `--verbose` prints debug logging to help diagnose convergence

A typical run prints ERLE statistics and convergence timing. Use `--help` for
the full CLI.

### Choosing Tap Length
For good convergence make sure the tap count spans the entire simulated echo
path.  With an echo delay of `delay_ms` and a tail of roughly `tail_taps`
additional samples, use:
```
taps >= sample_rate * delay_ms / 1000 + tail_taps
```
Example: `--delay-ms 32` with a 128-tap tail at 8 kHz needs ~384 taps.

## Module Layout
- `oslec_wrapper.py` – `ctypes` wrapper with helpers to convert PCM frames
- `signal_generator.py` – synthetic speech, tones, and noise builders
- `echo_simulator.py` – impulse-response modelling and noise injection
- `analyzer.py` – ERLE, convergence, and power metrics
- `test_suite.py` – configurable scenarios + optional plotting frontend

## Next Steps
- Add additional scenarios (fax tones, changing echo paths)
- Integrate PESQ or POLQA tooling for perceptual quality scoring
- Hook into automated sweeps for regression tracking
