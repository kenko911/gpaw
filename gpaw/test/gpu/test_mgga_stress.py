"""Meta-GGA stress must keep the CPU XC evaluator off GPU arrays."""

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress

from gpaw import GPAW
from gpaw.gpu import cupy, cupy_is_fake
from gpaw.new import xc as xc_module


def distorted_cell(symbol):
    atoms = (bulk('Si', 'diamond', a=5.43) if symbol == 'Si'
             else bulk('Al', 'fcc', a=4.05, cubic=True))
    atoms.set_cell(atoms.cell @ [[1.01, 0.007, 0],
                                [0, 0.995, 0.004],
                                [0, 0, 1.003]], scale_atoms=True)
    atoms.positions[0] += [0.04, -0.025, 0.015]
    return atoms


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.mgga
@pytest.mark.parametrize('xc', ['TPSS', 'MGGA_X_R2SCAN+MGGA_C_R2SCAN'])
@pytest.mark.parametrize('spinpol', [False, True])
@pytest.mark.skipif(cupy_is_fake, reason='Requires real GPU arrays')
def test_mgga_stress_cpu_gpu(xc, spinpol, monkeypatch, in_tmp_dir):
    atoms = distorted_cell('Si')
    if spinpol:
        atoms.set_initial_magnetic_moments([0.1, 0.1])

    gradient_and_sigma = xc_module.gradient_and_sigma

    def cpu_gradient(grad_v, density):
        # A backend mismatch otherwise reaches native NumPy code and segfaults.
        assert density.xp is np, 'Meta-GGA XC gradients require CPU arrays'
        return gradient_and_sigma(grad_v, density)

    monkeypatch.setattr(xc_module, 'gradient_and_sigma', cpu_gradient)
    predictions = []
    for gpu in [False, True]:
        atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 300},
                          xc=xc,
                          spinpol=spinpol,
                          kpts=(2, 2, 2),
                          occupations={'name': 'fermi-dirac', 'width': 0.05},
                          convergence={'energy': 1e-8, 'density': 1e-8,
                                       'eigenstates': 1e-10},
                          symmetry={'point_group': False},
                          parallel={'gpu': gpu},
                          setups='paw',
                          txt=f'{gpu}.txt')
        energy = atoms.get_potential_energy(force_consistent=True)
        forces = atoms.get_forces()
        assert atoms.calc.dft.ibzwfs.xp is (cupy if gpu else np)
        assert atoms.calc.dft.density.nt_sR.xp is (cupy if gpu else np)
        stress = atoms.get_stress()
        predictions.append((energy, forces, stress))
        atoms.calc = None

    cpu, gpu = predictions
    assert gpu[0] == pytest.approx(cpu[0], abs=1e-5 * len(atoms))
    assert gpu[1] == pytest.approx(cpu[1], abs=1e-4)
    assert gpu[2] == pytest.approx(cpu[2], abs=1e-5)


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.mgga
@pytest.mark.parametrize('symbol', ['Si', 'Al'])
@pytest.mark.skipif(cupy_is_fake, reason='Requires real GPU arrays')
def test_r2scan_gpu_stress_finite_difference(symbol, in_tmp_dir):
    atoms = distorted_cell(symbol)
    if symbol == 'Al':
        # At right angles the gradient stencil can switch under a shear,
        # making the discrete energy discontinuous even on CPU.
        atoms.set_cell(atoms.cell @ [[1.02, 0.017, 0.013],
                                    [0.011, 0.99, -0.019],
                                    [0.021, -0.014, 1.03]], scale_atoms=True)
    eps = 1e-5 if symbol == 'Al' else 1e-4
    atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 300},
                      xc='MGGA_X_R2SCAN+MGGA_C_R2SCAN',
                      kpts=(2, 2, 2),
                      occupations={'name': 'fermi-dirac', 'width': 0.05},
                      convergence={'energy': 1e-9, 'density': 1e-9,
                                   'eigenstates': 1e-11},
                      symmetry={'point_group': False},
                      parallel={'gpu': True},
                      setups='paw',
                      txt=f'{symbol}.txt')
    analytic = atoms.get_stress()
    numerical = calculate_numerical_stress(atoms, eps=eps,
                                         force_consistent=True)
    print(f'{symbol} analytic stress:', analytic)
    print(f'{symbol} numerical stress:', numerical)
    print(f'{symbol} maximum error:', abs(analytic - numerical).max())
    assert analytic == pytest.approx(numerical, abs=1e-4)
    assert atoms.calc.dft.ibzwfs.xp is cupy
    atoms.calc = None
