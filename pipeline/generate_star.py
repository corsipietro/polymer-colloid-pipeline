#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the LAMMPS data file ``star.dat`` for a star polymer (+ optional colloids).

Atom types:  1 = arm monomer, 2 = star core (anchor), 3 = colloids of species 1,
4 = colloids of species 2 (only the species that are present).
Bond types:  1 = monomer-monomer, 2 = core-first monomer of an arm.
The core sits at the box centre, the arms are straight rays whose directions
are drawn at random on the sphere with a minimum separation, and the colloids
are placed at random, not overlapping each other nor the initial star.

This module only holds the builder; ``generate_system.py`` is the entry point.
"""

import numpy as np

from functions import DEFAULT_RCORE


def random_arm_directions(arms, radius, min_dist, rng, max_tries=200000):
    """Draw ``arms`` unit vectors uniformly on the sphere, pairwise well separated.

    Random sequential insertion: a candidate direction is accepted only if the
    points ``radius * direction`` lie more than ``min_dist`` from all accepted
    ones.

    Args:
        arms: number of directions.
        radius: radius of the sphere on which the first monomers sit.
        min_dist: minimum allowed distance between first monomers.
        rng: ``numpy.random.RandomState``.
        max_tries: total number of candidates before giving up.

    Returns:
        Array of shape ``(arms, 3)``.

    Raises:
        RuntimeError: if the arms do not fit on the core within ``max_tries``.
    """
    directions = np.zeros((arms, 3))
    accepted = 0
    for _ in range(max_tries):
        if accepted == arms:
            break
        cos_theta = 2.0 * rng.random_sample() - 1.0  # uniform on the sphere
        phi = 2.0 * np.pi * rng.random_sample()
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        candidate = np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta])
        if accepted:
            gaps = radius * np.linalg.norm(directions[:accepted] - candidate, axis=1)
            if np.any(gaps <= min_dist):
                continue
        directions[accepted] = candidate
        accepted += 1
    if accepted != arms:
        raise RuntimeError(
            "could not place {} arms on a core of radius {} (placed {}): "
            "increase 'rcore' in par.txt".format(arms, radius, accepted))
    return directions


def place_colloids(colloids, box, clearance, rng, max_tries=100000):
    """Place the colloids at random positions in the periodic box.

    Each colloid must be farther than ``clearance + radius`` from the box centre (the
    star) and than the sum of the radii from every colloid already placed
    (minimum image convention).

    Args:
        colloids: list of ``(atom_type, count, diameter)``, one entry per species.
        box: ``[Lx, Ly, Lz]``.
        clearance: radius of the region around the centre occupied by the star.
        rng: ``numpy.random.RandomState``.
        max_tries: candidates tried per colloid before giving up.

    Returns:
        list of ``(atom_type, xyz)``.

    Raises:
        RuntimeError: if a colloid does not fit.
    """
    box = np.asarray(box, dtype=float)
    center = box / 2
    placed = []                                 # (atom_type, xyz, radius)

    def periodic_distance(a, b):
        """Minimum-image distance between two points."""
        d = a - b
        return np.linalg.norm(d - box * np.round(d / box))

    for atom_type, count, diameter in colloids:
        radius = diameter / 2
        for _ in range(count):
            for _ in range(max_tries):
                xyz = rng.uniform(0, 1, 3) * box
                if periodic_distance(xyz, center) <= clearance + radius:
                    continue
                if any(periodic_distance(xyz, p) <= radius + r for _, p, r in placed):
                    continue
                placed.append((atom_type, xyz, radius))
                break
            else:
                raise RuntimeError(
                    "could not place a colloid of diameter {} in the box (edge {}): "
                    "increase lbox".format(diameter, box[0]))
    return [(t, xyz) for t, xyz, _ in placed]


def build_star(arms, ns, sigma, box, rng, r_core=DEFAULT_RCORE, colloids=(), n_types=None):
    """Return the text of the LAMMPS data file for a star (and its colloids).

    Atom ids: 1 = core, 2..1+arms*ns = arm monomers (arm after arm, from the
    core outwards), then the colloids (species 1 first).

    Args:
        arms: number of arms.
        ns: monomers per arm.
        sigma: monomer diameter.
        box: ``[Lx, Ly, Lz]``.
        rng: ``numpy.random.RandomState``.
        r_core: core radius (``rcore`` in par.txt).
        colloids: list of ``(atom_type, count, diameter)`` per colloid species present.
        n_types: number of atom types to declare (default: 2 + number of species).
    """
    center = np.asarray(box, dtype=float) / 2
    ntot = 1 + ns * arms  # core + monomers
    natoms = ntot + sum(count for _, count, _ in colloids)
    if n_types is None:
        n_types = max([2] + [t for t, _, _ in colloids])
    directions = random_arm_directions(arms, r_core + sigma / 2, sigma, rng)
    positions = place_colloids(colloids, box, r_core + ns * sigma, rng)

    out = []
    out.append("LAMMPS data file\n\n")
    out.append("{} atoms\n".format(natoms))
    out.append("{} bonds\n".format(ntot - 1))
    out.append("0 angles\n0 dihedrals\n0 impropers\n\n")
    out.append("{} atom types\n2 bond types\n0 angle types\n0 dihedral types\n"
               "0 improper types\n\n".format(n_types))
    out.append("0.0 {0} xlo xhi\n0.0 {1} ylo yhi\n0.0 {2} zlo zhi\n\n".format(*box))
    out.append("Masses\n\n" + "".join("{} 1\n".format(t) for t in range(1, n_types + 1)) + "\n")

    # Atoms: "id molecule type x y z"
    out.append("Atoms\n\n")
    out.append("1 1 2 {0} {1} {2}\n".format(*center))
    arm_ids = []
    for arm, direction in enumerate(directions):
        ids = list(range(2 + arm * ns, 2 + (arm + 1) * ns))
        arm_ids.append(ids)
        for position, atom_id in enumerate(ids, 1):
            xyz = center + (r_core + sigma * position - sigma / 2) * direction
            out.append("{0} 1 1 {1} {2} {3}\n".format(atom_id, *xyz))
    for index, (atom_type, xyz) in enumerate(positions, ntot + 1):
        out.append("{0} 2 {1} {2} {3} {4}\n".format(index, atom_type, *xyz))
    out.append("\n")

    # Bonds: "id type atom1 atom2"
    out.append("Bonds\n\n")
    bond_id = 1
    for ids in arm_ids:                       # core - first monomer of each arm
        out.append("{0} 2 1 {1}\n".format(bond_id, ids[0]))
        bond_id += 1
    for ids in arm_ids:                       # monomer - monomer along each arm
        for first, second in zip(ids[:-1], ids[1:]):
            out.append("{0} 1 {1} {2}\n".format(bond_id, first, second))
            bond_id += 1
    out.append("\n")

    # Velocities are all zero here; LAMMPS thermalises the system at run time.
    out.append("Velocities\n\n")
    for atom_id in range(1, natoms + 1):
        out.append("{0} 0 0 0\n".format(atom_id))
    out.append("\n")
    return "".join(out)
