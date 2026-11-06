"""Tests for generate_star.py and generate_input.py."""
import os

import numpy as np
import pytest

import functions as fn
import generate_bottle_brush as gb
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


def test_star_without_colloid():
    """colloids 0: no colloid atom, two atom types, no colloid lines in the input."""
    text = gs.build_star(4, 3, 1.0, [40.0] * 3, np.random.RandomState(2), colloid=False)
    n_atoms, n_bonds, atoms, bonds = parse_data(text)
    assert n_atoms == 4 * 3 + 1 == len(atoms) and n_bonds == 12
    assert "2 atom types" in text and all(t != 3 for t, _ in atoms.values())
    params = fn.Params(arms=4, ns=3, lbox=40.0, colloids=0)
    inp = gi.build_input(params, 0, False)
    assert "type 3" not in inp and "pair_coeff 1 3" not in inp and "velocity coll" not in inp


def test_set_anchors_symmetric():
    """99 monomers, step 3: 33 anchors centred on the backbone; odd leftovers drop one monomer."""
    anchors, nb = gb.set_anchors(3, 99)
    assert nb == 99 and len(anchors) == 33 and anchors[0] == 2 and anchors[-1] == 98
    anchors, nb = gb.set_anchors(3, 98)
    assert nb == 97 and anchors[0] == 1 and anchors[-1] == 97
    assert gb.set_anchors(1, 10) == (list(range(1, 11)), 10)


def test_bottlebrush_data_file_is_consistent():
    """Counts, types, bond lengths and masses of the generated bottlebrush."""
    nb, ns, step = 99, 50, 3
    text = gb.build_bottle_brush(nb, ns, 1.0, step, [120.0] * 3, np.random.RandomState(1))
    n_atoms, n_bonds, atoms, bonds = parse_data(text)
    assert n_atoms == 1749 == len(atoms)           # same size as the original configurations
    assert n_bonds == 1748 == len(bonds)
    assert "4 atom types" in text and "\n3 1\n4 1\n" in text
    assert sum(1 for t, _ in atoms.values() if t == 2) == nb
    for _, kind, a, b in bonds:
        assert kind == 1
        assert np.linalg.norm(atoms[a][1] - atoms[b][1]) == pytest.approx(1.0)
    coords = np.array([xyz for _, xyz in atoms.values()])
    assert coords.min() >= 0 and coords.max() <= 120


def test_bottlebrush_box_too_small():
    """A box that cannot hold the straight brush is rejected."""
    with pytest.raises(ValueError, match="box too small"):
        gb.build_bottle_brush(99, 50, 1.0, 3, [100.0] * 3, np.random.RandomState(1))


def test_bbrush_input_matches_original_in_bbrush():
    """The generated in.bbrush equals the original, except for the WCA cutoff.

    The original used the rounded cutoff 1.1224; the generator uses 2^(1/6).
    """
    params = fn.read_par(os.path.join(HERE, "data", "par_bbrush.txt"))
    params.colloids = 1
    text = gi.build_input(params, 0, False, seeds=(9592497, 9336433))
    with open(os.path.join(HERE, "data", "in.bbrush.expected")) as handle:
        expected = handle.read()
    expected = expected.replace("pair_style lj/cut 1.1224\n",
                                "pair_style lj/cut 1.122462048309373\n")
    assert text == expected


def test_bbrush_input_restart_and_no_colloids():
    """Restart reads bbrush2.restart without create_atoms; no colloids drops types 3/4."""
    params = fn.read_par(os.path.join(HERE, "data", "par_bbrush.txt"))
    text = gi.build_input(params, 2, True)
    assert "read_restart bbrush2.restart remap" in text and "create_atoms" not in text
    assert "gyr_all_2.txt" in text and "bbrush_2.lammpstrj" in text
    bare = gi.build_input(params.replace(colloids=0), 0, False)
    assert "create_atoms" not in bare and "pair_coeff 1 3" not in bare and "group bbone id 1:99" in bare
