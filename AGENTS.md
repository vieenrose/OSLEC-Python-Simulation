# OSLEC - Open Source Line Echo Canceller

## Build Commands
- Build: `make` (builds kernel module using standard Linux kernel build system)
- Clean: `make clean`
- Install: `make install` (requires root privileges)
- Test: No dedicated test framework - use kernel module testing tools

## Code Style Guidelines

### General
- Linux kernel coding style (checkpatch.pl compliant)
- GPL v2 license headers required
- 8-space tabs for indentation
- 80-character line limit where practical

### Types & Naming
- Use kernel types: `int16_t`, `int32_t`, `uint16_t` for fixed-point DSP
- Function names: `module_function()` (snake_case)
- Variables: snake_case, descriptive names
- Constants: UPPER_CASE with descriptive names
- Struct names: `snake_case_t` suffix

### Imports & Headers
- Linux kernel headers first: `<linux/kernel.h>`, `<linux/module.h>`
- Local headers in quotes: `"echo.h"`, `"fir.h"`
- Include guards: `__FILENAME_H` format

### Error Handling
- Use kernel memory allocation: `kzalloc()`, `kcalloc()`
- Check allocation failures, cleanup properly
- Return NULL on creation failure, void for cleanup
- Use `EXPORT_SYMBOL_GPL()` for public functions

### DSP-Specific
- Fixed-point arithmetic (16-bit coefficients)
- Inline functions for performance-critical paths
- Platform-specific optimizations (#ifdef __bfin__)
- Proper scaling to prevent overflow/underflow