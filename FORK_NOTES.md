# GPAW meta-GGA GPU stress branch

This personal fork preserves GPAW's upstream history from
<https://gitlab.com/gpaw/gpaw>. Branch `r2scan-gpu-stress` starts at the
26.7.0 release commit `9c6f4ccd94355b3e8c1c418b4b605d7ce7552e30`.
`main` and tag `26.7.0` retain the unmodified upstream release.

## Fix

The unmodified release passed r2SCAN GPU energy and force calculations but
segfaulted when requesting stress for distorted silicon and aluminum cells.
Its CPU XC gradients received CuPy arrays. Other stress intermediates also
assumed CPU wavefunctions.

The branch converts interpolated XC stress inputs to NumPy, keeps the
kinetic-energy-density tensor on the wavefunction backend, and returns the
XC stress contribution on the plane-wave accumulator's backend. It retains
GPAW's CPU LibXC evaluation inside the GPU calculation and does not restart
SCF on CPU.

## Validation

Validation uses an NVIDIA H100, ASE 3.29.0, LibXC 7.0.0, CUDA 12.6.2, and
CuPy 13.4.1, with one process and one GPU:

- New real-GPU tests cover TPSS and r2SCAN CPU/GPU energy, forces, and stress,
  each with and without spin polarization.
- Independent finite-strain tests cover all six r2SCAN stress components
  for distorted Si2 and Al4 cells, with force-consistent energies.
- Existing silicon TPSS and spin-polarized iron revTPSS CPU stress reference
  tests check the existing CPU path.
- The original eight CPU/GPU PBE/r2SCAN calculations check both formerly
  crashing cells with the original 300 eV, 2×2×2 k-point settings. Native
  CUDA kernels, real CuPy, GPU wavefunctions, and GPU densities are required.

The nearly orthogonal aluminum cell switches its finite-difference gradient
stencil under two shear perturbations. CPU-only controls reproduced a finite
energy jump, so that cell is unsuitable for a strain-derivative test at those
settings. The aluminum finite-strain test uses a fully skewed cell with an
unchanged stencil across the perturbations. The original cell remains in the
CPU/GPU parity campaign. The stress-error tolerance was not relaxed.

All six new regression cases and the two existing CPU reference tests passed.
The original eight-calculation campaign passed energy, force, and stress
comparisons for both functionals and cells. The largest r2SCAN CPU/GPU stress
difference was 2.845e-9 eV/Å³. The maximum finite-strain stress errors were
1.442e-6 eV/Å³ for silicon and 7.145e-6 eV/Å³ for skewed aluminum, below the
unchanged 1e-4 eV/Å³ test tolerance.

Run the regression tests in a CUDA-enabled GPAW build:

```bash
python -m pytest gpaw/test/gpu/test_mgga_stress.py
python -m pytest gpaw/test/pw/test_si_stress_mgga.py gpaw/test/pw/test_fe_stress_mgga.py
```

These checks establish backend agreement and strain derivatives for the
tested cases. They do not establish DFT cutoff/k-point convergence, speedup,
distributed MPI behavior, or behavior on other GPU architectures.

## Build

GPAW 26.7 requires ASE 3.29. Use a separate environment from applications
pinning older ASE versions. Follow `doc/documentation/gpu.rst` to build with
the required CUDA or HIP toolchain, real CuPy, BLAS, and LibXC.

Build from the Git checkout: the published 26.7.0 source archive omitted
CUDA C++ headers present in the upstream repository. The validation used
an MPI-enabled CUDA/C++ build, with CUDA include paths configured for both
the host compiler and nvcc. The normal upstream license remains in effect.
