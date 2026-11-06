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
    text = gs.build_star(arms, ns, sigma, box, np.random.RandomState(1), colloids=[(3, 1, 6.0)])
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
    text = gs.build_star(4, 3, 1.0, [40.0] * 3, np.random.RandomState(2), n_types=2)
    n_atoms, n_bonds, atoms, bonds = parse_data(text)
    assert n_atoms == 4 * 3 + 1 == len(atoms) and n_bonds == 12
    assert "2 atom types" in text and all(t != 3 for t, _ in atoms.values())
    params = fn.Params(arms=4, ns=3, lbox=40.0, colloids=0)
    inp = gi.build_input(params, 0, False)
    assert "pair_style lj/sf" in inp and "hybrid" not in inp
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
    params = fn.read_par(os.path.join(HERE, "data", "par_bbrush.txt"))    # two colloid species
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


def test_bbrush_colloid_size_from_q():
    """With q, species 1 has diameter 2 rc / q (rc = effective-star radius); species 2 keeps dcoll2."""
    params = fn.read_par(os.path.join(HERE, "data", "par_bbrush.txt")).replace(q=2.5, ncoll1=1, ncoll2=0)
    rc = gb.effective_star_radius(99, 50, 3)
    assert rc == pytest.approx(0.5 * 50 ** 0.588 * (50 / 3) ** (0.588 / 5) * 1.0, rel=1e-9)
    d1, d2 = gb.colloid_diameters(params)
    assert d1 == pytest.approx(2 * rc / 2.5) and d2 == 10.0
    text = gi.build_input(params, 0, False, seeds=(1, 2))
    assert "create_atoms 3 random 1 1 NULL" in text and "create_atoms 4" not in text
    assert "pair_coeff 3 3 1.0 {0} ".format(d1) in text
    bigger = gb.colloid_diameters(params.replace(q=1.25))[0]
    assert bigger == pytest.approx(2 * d1)       # halving q doubles the colloid


def star_params(**kw):
    """Small star run with colloids of size ratios q=1.8 and q2=2.6."""
    base = dict(arms=6, ns=10, lbox=80.0, q=1.8, q2=2.6, eps=0.2, colloids=2, ncoll1=2, ncoll2=3)
    base.update(kw)
    return fn.Params(**base)


def test_star_with_two_species_data_file():
    """Two colloid species: types 3 and 4, right counts, no overlaps, clear of the star."""
    params = star_params()
    d1, d2 = fn.star_colloid_diameters(params)
    assert d1 == pytest.approx(2 * fn.star_radius(6, 10) / 1.8) and d2 > 0
    groups = [(t, n, (d1, d2)[i - 1]) for i, t, n in params.species()]
    text = gs.build_star(6, 10, 1.0, params.box, np.random.RandomState(4), colloids=groups, n_types=4)
    n_atoms, n_bonds, atoms, _ = parse_data(text)
    assert n_atoms == 6 * 10 + 1 + 5 and "4 atom types" in text
    colloids = [(t, xyz) for t, xyz in atoms.values() if t >= 3]
    assert [t for t, _ in colloids].count(3) == 2 and [t for t, _ in colloids].count(4) == 3
    box = np.array(params.box)
    radius = {3: d1 / 2, 4: d2 / 2}
    centre = box / 2

    def dist(a, b):
        """Minimum-image distance."""
        d = a - b
        return np.linalg.norm(d - box * np.round(d / box))

    for i, (t, xyz) in enumerate(colloids):
        assert dist(xyz, centre) > fn.DEFAULT_RCORE + 10 + radius[t]
        for t2, xyz2 in colloids[:i]:
            assert dist(xyz, xyz2) > radius[t] + radius[t2]


def test_colloid_placement_fails_in_a_tiny_box():
    """An impossible placement raises instead of looping forever."""
    with pytest.raises(RuntimeError, match="lbox"):
        gs.place_colloids([(3, 1, 30.0)], [40.0] * 3, 20.0, np.random.RandomState(0), max_tries=500)


def test_star_input_one_and_two_species():
    """One species writes only type 3 coefficients; two add type 4 and the 3-4 cross term."""
    one = gi.build_input(star_params(colloids=1), 0, False)
    assert "pair_coeff 1 3 lj/cut" in one and "pair_coeff 1 4" not in one and "group coll type 3\n" in one
    two = gi.build_input(star_params(), 0, False)
    for line in ("pair_coeff 1 4 lj/cut", "pair_coeff 2 4 lj/cut", "pair_coeff 4 4 lj/sf", "pair_coeff 3 4 lj/sf"):
        assert line in two
    assert "group coll type 3 4" in two
    d1, d2 = fn.star_colloid_diameters(star_params())
    assert "pair_coeff 3 4 lj/sf 1.0 {0} ".format(np.mean([d1, d2])) in two
    # a second eps for species 2
    other = gi.build_input(star_params(eps2=0.4), 0, False)
    assert "pair_coeff 1 4 lj/cut 0.4 " in other and "pair_coeff 1 3 lj/cut 0.2 " in other


def test_bbrush_one_species_and_polymer_alone():
    """colloids 1 -> only species 1 (type 3); colloids 0 -> no colloid lines at all."""
    base = fn.read_par(os.path.join(HERE, "data", "par_bbrush.txt"))
    one = gi.build_input(base.replace(colloids=1, ncoll1=5), 0, False, seeds=(1, 2))
    assert "create_atoms 3 random 5 1 NULL" in one and "create_atoms 4" not in one
    assert "pair_coeff 3 3" in one and "pair_coeff 4 4" not in one and "pair_coeff 3 4" not in one
    alone = gi.build_input(base.replace(colloids=0), 0, False)
    assert "create_atoms" not in alone and "pair_coeff 1 3" not in alone
    assert "2 atom types" in gb.build_bottle_brush(30, 5, 1.0, 3, [60.0] * 3, np.random.RandomState(0), n_types=2)
