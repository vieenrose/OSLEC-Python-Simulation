# Benchmark Hyperparameters

## Default Sweep Ranges
- **NLMS**: `mu ∈ {0.6, 0.8, 1.0}`
- **IPNLMS**: `(mu, alpha) ∈ {(0.5, 0.3), (0.6, 0.5), (0.7, 0.6)}`
- **MDF**: `(block_size, mu) ∈ {(128, 0.4), (128, 0.5), (256, 0.35)}`
- **OSLEC**: Uses module defaults (no sweep)

The benchmark selects the configuration with the highest mean ERLE per scenario
and prints the chosen values for traceability. Use `--plot` to render the
microphone, residual, and ERLE traces with hyperparameters shown in the title.
