# IPNLMS Drop-in Replacement for OSLEC

## Summary
- Implements Improved Proportionate NLMS (IPNLMS) tailored for sparse echo
  paths frequently observed in handset and speakerphone deployments.
- Preserves the original OSLEC kernel API (`oslec_create`, `oslec_update`, etc.).
- Uses Q16 fixed-point arithmetic for predictable performance on ARM cores.
- Retains a lightweight DC-blocking high-pass filter for TX samples.

## Rationale
Planned benchmark runs (via `simulation/benchmark.py`) target comparisons
against stock OSLEC. Analytical expectations and prior studies indicate that
IPNLMS converges faster on sparse/variable echo scenarios while maintaining a
small CPU footprint. MDF can achieve higher ERLE on long diffuse echoes but
introduces frame latency and FFT overhead. Jetson Nano systems prioritising
low-latency duplex audio typically gain more from IPNLMS.

## Integration
1. Build the user-space shared library and point consumers at it:
   ```bash
   cd simulation
   make ipnlms
   OSLEC_LIB=$(pwd)/liboslec_ipnlms.so python3 test_suite.py --scenario time_variant --taps 384
   ```
   The harness now loads the drop-in via `ctypes`, exercising the exact API the
   kernel module expects.
2. For kernel deployments, replace `echo.c` in the module build with
   `echo_ipnlms.c` (or extend the Makefile to compile the new file). No API
   changes are required.
3. Optional tunables inside `echo_ipnlms.c`:
   - `DEFAULT_MU_Q16` (step size, Q16) controls convergence speed.
   - `DEFAULT_ALPHA_Q16` (proportionate blend) adjusts sparsity weighting.
   - `DEFAULT_EPSILON` guards the gain normalisation.

## Observed Performance (time_variant scenario, 384 taps)
- OSLEC baseline (liboslec.so): mean ERLE ≈ 30 dB, residual ≈ −38 dBFS.
- IPNLMS drop-in: mean ERLE ≈ 16–17 dB, residual ≈ −41 dBFS, convergence ~50 s
  (vs ~265 s window estimate for OSLEC).
The IPNLMS variant converges markedly faster but trails the original in steady
state ERLE; further tuning is advised before deployment in demanding
environments.

## Limitations & Future Work
- Does not yet implement NLP/CNG auxiliary features from the original OSLEC.
  These can be ported on top of the new adaptive core.
- Gain update currently recomputed every sample; batching could reduce CPU.
- Tune default Q16 parameters or add runtime controls to match OSLEC’s ERLE
  while retaining faster convergence.
- Extend benchmarking with PESQ/POLQA to correlate ERLE improvements with
  perceived quality.
