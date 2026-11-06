#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the LAMMPS data file ``bbrush.dat`` for a bottlebrush.

Atom types:  1 = side-chain monomer, 2 = backbone monomer, and (only if
colloids are present) 3 and 4 = the two colloid species. The colloids are *not*
written to the data file: ``in.bbrush`` inserts them with ``create_atoms``, but
their types and masses are declared here so that LAMMPS accepts them.
Bond type 1 (FENE) is used everywhere.

The backbone is a straight line along y through the box centre; one side
chain of ``ns`` monomers is grafted every ``step`` backbone monomers, pointing
radially in a random direction in the xz plane. This module only holds the
builder; ``generate_system.py`` is the entry point.
"""

import numpy as np


def set_anchors(step, nb):
    """Return ``(anchors, nb)``: the backbone ids that carry a side chain.

    Anchors are spaced by ``step`` and centred along the backbone. If the
    leftover monomers cannot be split evenly between the two ends, one backbone
    monomer is dropped, so the returned ``nb`` can be ``nb - 1``.
    """
    if step == 1:
        anchors = list(range(1, nb + 1))
    else:
        anchors = list(range(1, nb, step))

    lone_monomers = nb - max(anchors)
    if lone_monomers != 0:
        if lone_monomers % 2 == 0:
            anchors = [i + lone_monomers // 2 for i in anchors]
        else:
            nb -= 1
            lone_monomers -= 1
            if lone_monomers != 0:
                anchors = [i + lone_monomers // 2 for i in anchors]
    return anchors, nb


def effective_star_radius(nb, ns, step):
    """Return the radius ``rc`` of the effective star of the brush.

    ``rc = 0.5 * ns**0.588 * feq**(1/5)`` with ``feq = (g * ns)**0.588`` and ``g``
    the grafting density (side chains per backbone monomer).
    """
    nb, nbrush, _ = brush_size(nb, ns, step)
    feq = (nbrush / nb * ns) ** 0.588
    return 0.5 * ns ** 0.588 * feq ** (1 / 5)


def colloid_diameters(params):
    """Return the diameters ``(d1, d2)`` of the two colloid species.

    A species whose size ratio ``q`` (``q`` for species 1, ``q2`` for species 2) is
    set has diameter ``2 * rc / q`` (``rc`` = effective-star radius, so ``q`` is the
    polymer/colloid size ratio); otherwise the explicit ``dcoll1`` / ``dcoll2`` are used.
    """
    rc = effective_star_radius(params.nb, params.ns, params.step)
    d1 = 2 * rc / params.q if params.q is not None else params.dcoll1
    d2 = 2 * rc / params.q2 if params.q2 is not None else params.dcoll2
    return d1, d2


def check_box(nb, ns, sigma, box):
    """Raise ValueError if the straight brush does not fit in the periodic box."""
    if box[1] <= (nb + 2) * sigma:
        raise ValueError("box too small along the backbone: need > {}, got {}".format(
            (nb + 2) * sigma, box[1]))
    if box[0] <= 2 * (ns + 1) * sigma:
        raise ValueError("box too small for the side chains: need > {}, got {}".format(
            2 * (ns + 1) * sigma, box[0]))


def brush_size(nb, ns, step):
    """Return ``(nb, nbrush, natoms)`` of the brush (without colloids)."""
    anchors, nb = set_anchors(step, nb)
    return nb, len(anchors), nb + ns * len(anchors)


def build_bottle_brush(nb, ns, sigma, step, box, rng, colloids=True):
    """Return the text of the LAMMPS data file of a bottlebrush.

    Atom ids: 1..nb = backbone, then the side chains one after the other
    (each from the anchor outwards).

    Args:
        nb: backbone monomers requested (see :func:`set_anchors`).
        ns: monomers per side chain.
        sigma: monomer diameter.
        step: backbone monomers between two anchors.
        box: ``[Lx, Ly, Lz]``.
        rng: ``numpy.random.RandomState`` (side-chain directions).
        colloids: declare the colloid types 3 and 4.
    """
    anchors, nb = set_anchors(step, nb)
    check_box(nb, ns, sigma, box)
    nbrush = len(anchors)
    ntot = nb + ns * nbrush
    center = np.asarray(box, dtype=float) / 2

    out = []
    out.append("LAMMPS data file\n\n")
    out.append("{0} atoms\n{1} bonds\n".format(ntot, ntot - 1))
    out.append("0 angles\n0 dihedrals\n0 impropers\n\n")
    ntypes = 4 if colloids else 2
    out.append("{} atom types\n1 bond types\n0 angle types\n0 dihedral types\n"
               "0 improper types\n\n".format(ntypes))
    out.append("0.0 {0} xlo xhi\n0.0 {1} ylo yhi\n0.0 {2} zlo zhi\n\n".format(*box))
    out.append("Masses\n\n" + "".join("{} 1\n".format(t) for t in range(1, ntypes + 1)) + "\n")

    out.append("Atoms\n\n")
    axis = np.array([0.0, 1.0, 0.0])
    start = center - axis * sigma * round(nb / 2)
    backbone = {}
    for j in range(1, nb + 1):
        backbone[j] = start + j * sigma * axis
        out.append("{0} 1 2 {1} {2} {3}\n".format(j, *backbone[j]))

    side_ids = {}
    next_id = nb + 1
    for anchor in anchors:
        ids = list(range(next_id, next_id + ns))
        next_id += ns
        side_ids[anchor] = ids
        angle = 2 * np.pi * rng.random_sample()
        direction = np.array([-np.sin(angle), 0.0, np.cos(angle)])
        for position, atom_id in enumerate(ids, 1):
            xyz = backbone[anchor] + position * sigma * direction
            out.append("{0} 1 1 {1} {2} {3}\n".format(atom_id, *xyz))
    out.append("\n")

    out.append("Velocities\n\n")
    for atom_id in range(1, ntot + 1):
        out.append("{0} 0 0 0\n".format(atom_id))
    out.append("\n")

    out.append("Bonds\n\n")
    bond_id = 1
    for j in range(1, nb):
        out.append("{0} 1 {1} {2}\n".format(bond_id, j, j + 1))
        bond_id += 1
    for anchor in anchors:
        chain = [anchor] + side_ids[anchor]
        for first, second in zip(chain[:-1], chain[1:]):
            out.append("{0} 1 {1} {2}\n".format(bond_id, first, second))
            bond_id += 1
    out.append("\n")
    return "".join(out)
