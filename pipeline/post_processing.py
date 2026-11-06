#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse the trajectories of one run and write histograms to ``analysis/``.

Reads ``configurations/star_<n>.lammpstrj`` (n = 0, 1, ... until a file is
missing) and ``par.txt``. The first frame of every file is skipped: it is the
configuration at the start of the (re)started run, not an equilibrated one.

Output files (two columns ``x y``, bin centre and normalised histogram):
  mon_distr_anch        monomer distance from the core
  mon_distr_cm          monomer distance from the centre of mass of the monomers
  coll_distr_anch       colloid distance from the core
  coll_distr_anch_2rg   same, only for frames where the colloid is within 2 rs
  coll_distr_cm         colloid distance from the centre of mass of the monomers
  rg                    distribution of the radius of gyration of the star
  loading.txt           per frame: colloid within rs, colloid within 2 rs (0/1)

``rs`` is the star radius ``0.6 * f**(1/5) * ns**0.588``. Distances use the
minimum image convention relative to the core.

Usage (from a run directory)::

    python3 post_processing.py
"""

import os

import numpy as np

from functions import create_dir, read_par, star_radius

# Lines of the header that precedes the atoms of every LAMMPS dump frame.
HEADER_LINES = 9

TYPE_MONOMER, TYPE_CORE, TYPE_COLLOID = 1, 2, 3


def read_frames(path, natoms, skip_first=True):
    """Yield the frames of a ``.lammpstrj`` file as ``(natoms, 5)`` arrays.

    Columns are ``id type x y z``. A truncated last frame (e.g. from a job
    killed while writing) is silently dropped.

    Raises:
        ValueError: if a frame declares a number of atoms different from
            ``natoms``.
    """
    with open(path) as handle:
        first = True
        while True:
            header = [handle.readline() for _ in range(HEADER_LINES)]
            if not header[-1]:
                return
            declared = int(header[3])
            if declared != natoms:
                raise ValueError("{}: frame has {} atoms, expected {}".format(
                    path, declared, natoms))
            rows = []
            for _ in range(natoms):
                line = handle.readline()
                if not line:
                    return
                rows.append(line.split())
            if first and skip_first:
                first = False
                continue
            first = False
            yield np.array(rows, dtype=float)


def frame_observables(frame, lbox):
    """Compute the per-frame quantities used by the histograms.

    Args:
        frame: ``(natoms, 5)`` array ``id type x y z``.
        lbox: edge of the cubic periodic box.

    Returns:
        dict with arrays ``mon_anch``, ``mon_cm`` (monomer distances),
        ``coll_anch``, ``coll_cm`` (colloid distances) and the float ``rg``.
    """
    types = frame[:, 1].astype(int)
    xyz = frame[:, 2:5]
    anchor = xyz[types == TYPE_CORE][0]

    # Unwrap every atom to its periodic image closest to the core.
    delta = xyz - anchor
    delta -= lbox * np.round(delta / lbox)
    xyz = anchor + delta

    monomers = xyz[types == TYPE_MONOMER]
    colloid = xyz[types == TYPE_COLLOID]

    # The centre of mass uses the monomers only (core excluded).
    cm = monomers.mean(axis=0)

    # Radius of gyration of the whole star (core + monomers), the same
    # definition as LAMMPS ``compute gyration`` on group "type 1 2".
    star = xyz[(types == TYPE_MONOMER) | (types == TYPE_CORE)]
    rg = np.sqrt(np.mean(np.sum((star - star.mean(axis=0)) ** 2, axis=1)))

    return {
        "mon_anch": np.linalg.norm(monomers - anchor, axis=1),
        "mon_cm": np.linalg.norm(monomers - cm, axis=1),
        "coll_anch": np.linalg.norm(colloid - anchor, axis=1),
        "coll_cm": np.linalg.norm(colloid - cm, axis=1),
        "rg": rg,
    }


class Histogram(object):
    """Average of per-frame normalised histograms on fixed, uniform bins."""

    def __init__(self, upper, nbins):
        """Create ``nbins`` bins covering ``[0, upper]``."""
        self.bins = np.linspace(0, upper, nbins + 1)
        self.total = np.zeros(nbins)
        self.frames = 0

    def add(self, values):
        """Add one frame. Frames with no value inside the bins are ignored.

        (``numpy.histogram(..., density=True)`` returns NaN for an empty
        histogram, which would corrupt the average.)
        """
        counts = np.histogram(values, bins=self.bins)[0]
        if counts.sum() == 0:
            return
        self.total += counts / (counts.sum() * np.diff(self.bins))
        self.frames += 1

    @property
    def x(self):
        """Bin centres."""
        return 0.5 * (self.bins[:-1] + self.bins[1:])

    @property
    def y(self):
        """Mean of the normalised per-frame histograms."""
        return self.total / max(self.frames, 1)


def analyse(config_dir, params):
    """Process all trajectory files of a run.

    Returns:
        ``(histograms, loading)``: dict name -> :class:`Histogram`, and a list
        of ``(within_rs, within_2rs)`` tuples, one per frame.

    Raises:
        RuntimeError: if no frame could be read.
    """
    lbox = params.box[0]
    rs = star_radius(params.arms, params.ns)
    natoms = params.arms * params.ns + 2  # monomers + core + colloid
    nbins_coll = int(round(lbox * 2))

    hist = {
        "mon_distr_anch": Histogram(3 * rs, 150),
        "mon_distr_cm": Histogram(3 * rs, 150),
        "coll_distr_anch": Histogram(lbox, nbins_coll),
        "coll_distr_anch_2rg": Histogram(lbox, nbins_coll),
        "coll_distr_cm": Histogram(lbox, nbins_coll),
    }
    rg_hist = Histogram(2 * rs, 200)
    rgs = []
    loading = []

    number = 0
    while True:
        path = os.path.join(config_dir, "star_{}.lammpstrj".format(number))
        if not os.path.isfile(path):
            break
        number += 1
        for frame in read_frames(path, natoms):
            obs = frame_observables(frame, lbox)
            hist["mon_distr_anch"].add(obs["mon_anch"])
            hist["mon_distr_cm"].add(obs["mon_cm"])
            hist["coll_distr_cm"].add(obs["coll_cm"])
            hist["coll_distr_anch"].add(obs["coll_anch"])
            hist["coll_distr_anch_2rg"].add(obs["coll_anch"][obs["coll_anch"] <= 2 * rs])
            rgs.append(obs["rg"])
            loading.append((int(np.sum(obs["coll_anch"] <= rs)),
                            int(np.sum(obs["coll_anch"] <= 2 * rs))))

    if not rgs:
        raise RuntimeError("no frames found in {}".format(config_dir))

    rg_hist.add(rgs)
    hist["rg"] = rg_hist
    return hist, loading


def write_results(hist, loading, out_dir="analysis"):
    """Write one ``<name>.txt`` per histogram plus ``loading.txt``."""
    create_dir(out_dir)
    for name, h in hist.items():
        with open(os.path.join(out_dir, name + ".txt"), "w") as handle:
            handle.write("# x    y\n")
            for x, y in zip(h.x, h.y):
                handle.write("{0} {1}\n".format(x, y))
    with open(os.path.join(out_dir, "loading.txt"), "w") as handle:
        handle.write("# within_rs within_2rs\n")
        for within_rs, within_2rs in loading:
            handle.write("{0} {1}\n".format(within_rs, within_2rs))


def main():
    """Analyse the run in the current directory."""
    params = read_par()
    hist, loading = analyse("configurations", params)
    write_results(hist, loading)


if __name__ == "__main__":
    main()
