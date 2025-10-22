# Lightweight Acoustic Echo Cancellation (AEC) Options for Jetson Nano

## Context
- Target hardware: NVIDIA Jetson Nano 4 GB (quad Cortex-A57 @ 1.43 GHz, 128 Maxwell CUDA cores, 4 GB LPDDR4).
- Application: Full-duplex telephony / conferencing with challenging echo paths (time-varying delays, double talk, tones, noise bursts).
- Baseline: OSLEC dual-path NLMS (C kernel module). Strengths: open-source, integer friendly, but struggles with long/variant echoes and aggressive noise.

## Candidate Algorithms

### 1. Partitioned Block Frequency-Domain NLMS (MDF / PBFDLMS)
- **Principle**: Partition adaptive filter into `P` blocks (`L` samples each), operate in frequency domain with FFT/overlap-save. Reduces per-sample complexity to `O(P log L)` and handles long impulse responses efficiently.
- **Implementations**: Speex AEC, older WebRTC AEC (AEC2) front-end.
- **Pros**:
  - Excellent for >128 tap paths; fast convergence on time-varying delays with block updates.
  - FFT ops map well to NEON and to CUDA (if offload desired).
  - Stable under double-talk with proper power estimates.
- **Cons**:
  - Requires buffering (`L`-sample latency).
  - Needs reliable noise/near-end detection to suspend adaptation.
- **Suitability**: Highly suitable. Jetson Nano can sustain 8 kHz–48 kHz audio with modest CPU.

### 2. Improved Proportionate NLMS (IPNLMS)
- **Principle**: Weight the adaptation step per tap to favor sparse impulse responses. IPNLMS blends standard NLMS with Proportionate NLMS for robustness.
- **Pros**:
  - Rapid convergence for sparse echoes (handsets, speakerphones).
  - Sample-domain; minimal latency; integer-friendly.
- **Cons**:
  - Less efficient for dense room responses (>10 ms sustain).
  - Needs careful step-size tuning to avoid instability.
- **Suitability**: Good fit where echo path energy is cluster/sparse (typical telephony). Complexity only ~15–20 % above NLMS.

### 3. Frequency-Domain Block LMS with Adaptive Step (FDBLMS-AS)
- **Principle**: Block LMS in frequency domain with adaptive step-size derived from error-to-reference power ratio. Similar to MDF but simpler partitioning and lower memory.
- **Pros**:
  - Fewer control paths than MDF; still handles long paths.
  - Adaptive step improves resilience to double-talk without explicit detector.
  - Amenable to NEON/CUDA acceleration.
- **Cons**:
  - Slightly slower than MDF under rapid echo changes.
  - Requires overlap-save buffering; moderate latency (`L` samples).
- **Suitability**: Balanced option when MDF’s full feature set is unnecessary.

### Honorable Mentions
- **WebRTC AEC3**: Very robust but heavier (dual echo model, nonlinear processing, delay estimator); borderline for Nano unless heavily optimised.
- **RNNoise-based suppression**: Good for background noise, not an echo canceller.
- **Recursive Least Squares (RLS)** variants: Fast convergence but prohibitive `O(N^2)` cost for long filters.

## Selected Algorithms for Evaluation
| Algorithm | Rationale | Expected Strengths |
|-----------|-----------|--------------------|
| **MDF (Partitioned Block FDNLMS)** | Industry-proven, handles long/variant paths | Fast convergence, good under double-talk with proper gating |
| **IPNLMS** | Lightweight sparse-echo specialist | Low latency, better than NLMS on handset-like echoes |
| **FDBLMS-AS** | Simplified block frequency approach | Balanced complexity, fewer heuristics |

These three cover complementary design points (frequency domain vs sample domain, sparse vs dense echo handling).

## Evaluation Metrics
- **ERLE trajectory** (per scenario) and steady-state average.
- **Convergence time** to reach target ERLE (20 dB and 35 dB thresholds).
- **Residual echo PSD** vs OSLEC.
- **Computational cost** (MACs/sample, FFT count per second).
- **Memory footprint** (coefficients, histories).

## Implementation Considerations
- Use 16-bit fixed-point where practical (Jetson NEON optimisations), keep floating reference model for prototyping.
- Share scenario generator with OSLEC simulation to maintain apples-to-apples comparison.
- Ensure algorithms expose the OSLEC API when porting to kernel (create/adaption/update/free).

