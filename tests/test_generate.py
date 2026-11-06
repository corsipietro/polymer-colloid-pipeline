"""Tests for generate_star.py and generate_input.py."""
import os

import numpy as np
import pytest

import functions as fn
import generate_input as gi
import generate_star as gs

HERE = os.path.dirname(__file__)


def parse_data(text):
    """Parse a LAMMPS data file into header counts, atoms and bonds."""
    lines = text.split("\n")
    n_atoms = int(lines[2].split()[0])
    n_bonds = int(lines[3].split()[0])
    atoms, bonds = {}, []
    section = None
    for line in lines:
        if line in ("Atoms", "Bonds", "Velocities", "Masses"):
            section = line
            continue
        parts = line.split()
        if not parts:
            continue
        if section == "Atoms":
            atoms[int(parts[0])] = (int(parts[2]), np.array(parts[3:6], float))
        elif section == "Bonds":
            bonds.append(tuple(int(p) for p in parts))
    return n_atoms, n_bonds, atoms, bonds


def test_star_data_file_is_consistent():
    """Counts, bond topology and bond lengths of a small star."""
    arms, ns, sigma, box = 6, 5, 1.0, [50.0] * 3
    text = gs.build_star(arms, ns, sigma, box, np.random.RandomState(1))
    n_atoms, n_bonds, atoms, bonds = parse_data(text)

    assert n_atoms == arms * ns + 2 == len(atoms)
    assert n_bonds == arms * ns == len(bonds)
    assert sum(1 for t, _ in atoms.values() if t == 1) == arms * ns
    assert atoms[1][0] == 2 and atoms[n_atoms][0] == 3

    for _, kind, a, b in bonds:
        length = np.linalg.norm(atoms[a][1] - atoms[b][1])
        expected = fn.DEFAULT_RCORE + sigma / 2 if kind == 2 else sigma
        assert length == pytest.approx(expected)
    assert sorted(b[0] for b in bonds) == list(range(1, n_bonds + 1))


def test_arm_directions_respect_min_distance():
    """First monomers of different arms are further apart than sigma."""
    radius = fn.DEFAULT_RCORE + 0.5
    d = gs.random_arm_directions(30, radius, 1.0, np.random.RandomState(0))
    gaps = [radius * np.linalg.norm(d[i] - d[j]) for i in range(30) for j in range(i)]
    assert min(gaps) > 1.0
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_arm_directions_fail_instead_of_looping_forever():
    """An impossible request raises RuntimeError."""
    with pytest.raises(RuntimeError):
        gs.random_arm_directions(500, fn.DEFAULT_RCORE + 0.5, 1.0, np.random.RandomState(0),
                                 max_tries=2000)


def test_large_star_fits_with_larger_core():
    """f=50 does not fit on the default core but fits with rcore=2.25."""
    with pytest.raises(RuntimeError, match="rcore"):
        gs.random_arm_directions(50, 1.75 + 0.5, 1.0, np.random.RandomState(0), max_tries=30000)
    d = gs.random_arm_directions(50, 2.25 + 0.5, 1.0, np.random.RandomState(0))
    assert d.shape == (50, 3)


def test_input_matches_reference_in_star():
    """in.star of the f=10, ns=50, q=1.8, eps=0.2 restart run is reproduced exactly."""
    params = fn.read_par(os.path.join(HERE, "data", "par_f10_ns50.txt"))
    # the reference file was written by a restarted run: output index 1
    text = gi.build_input(params, file_number=1, continue_run=True)
    with open(os.path.join(HERE, "data", "in.star.expected")) as handle:
        assert text == handle.read()


def test_input_fresh_run_reads_data_file():
    """A fresh run reads star.dat and uses output index 0."""
    params = fn.read_par(os.path.join(HERE, "data", "par_f10_ns50.txt"))
    text = gi.build_input(params, file_number=0, continue_run=False)
    assert "read_data star.dat" in text and "read_restart" not in text
    assert "gyr_all_0.txt" in text and "star_0.lammpstrj" in text
