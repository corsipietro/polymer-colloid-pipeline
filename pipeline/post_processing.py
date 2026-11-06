#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse the trajectories of one run and write histograms to ``analysis/``.

Reads ``configurations/<prefix>_<n>.lammpstrj`` (prefix ``star`` or ``bbrush``,
n = 0, 1, ... until a file is missing) and ``par.txt``. The first frame of every
file is skipped: it is the configuration at the start of the (re)started run,
not an equilibrated one.

STAR output files (two columns ``x y``, bin centre and normalised histogram;
the colloid files are written only if the run has a colloid):
  mon_distr_anch        monomer distance from the core
  mon_distr_cm          monomer distance from the centre of mass of the monomers
  coll_distr_anch       colloid distance from the core
  coll_distr_anch_2rg   same, only for frames where the colloid is within 2 rs
  coll_distr_cm         colloid distance from the centre of mass of the monomers
  rg                    distribution of the radius of gyration of the star
  loading.txt           per frame: colloid within rs, colloid within 2 rs (0/1)

``rs`` is the star radius ``0.6 * f**(1/5) * ns**0.588``. Distances use the
minimum image convention relative to the core.

BOTTLEBRUSH output files (new analysis, not the original one, which is lost):
  end_to_end.txt        per frame: backbone end-to-end distance, backbone Rg, brush Rg
  rg                    distribution of the brush radius of gyration
  mon_distr_cm          bead distance from the centre of mass of the brush
  coll_a_distr_cm       distance of the colloids of species 1 (type 3) from it
  coll_b_distr_cm       same for species 2 (type 4)
Positions are unwrapped around the backbone centre of mass, which assumes that
the brush is smaller than half the box.

Usage (from a run directory)::

    python3 post_processing.py
"""

import os

import numpy as np

from functions import create_dir, read_par, star_radius

# Lines of the header that precedes the atoms of every LAMMPS dump frame.
HEADER_LINES = 9

TYPE_MONOMER, TYPE_CORE, TYPE_COLLOID = 1, 2, 3          # star
TYPE_BB_SIDE, TYPE_BB_BACKBONE, TYPE_BB_COLL_A, TYPE_BB_COLL_B = 1, 2, 3, 4  # bottlebrush


def read_frames(path, natoms=None, skip_first=True):
    """Yield the frames of a ``.lammpstrj`` file as ``(natoms, 5)`` arrays.

    Columns are ``id type x y z``. A truncated last frame (e.g. from a job
    killed while writing) is silently dropped.

    Args:
        path: ``.lammpstrj`` file.
        natoms: expected number of atoms, or None to accept what the frame declares.
        skip_first: drop the first frame of the file.

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
            if natoms is None:
                count = declared
            else:
                count = natoms
            if declared != count:
                raise ValueError("{}: frame has {} atoms, expected {}".format(
                    path, declared, natoms))
            rows = []
            for _ in range(count):
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


def trajectory_frames(config_dir, prefix, natoms=None):
    """Yield all frames of ``<config_dir>/<prefix>_<n>.lammpstrj`` for n = 0, 1, ...

    Raises:
        RuntimeError: if the first file does not exist.
    """
    number = 0
    while True:
        path = os.path.join(config_dir, "{}_{}.lammpstrj".format(prefix, number))
        if not os.path.isfile(path):
            break
        number += 1
        for frame in read_frames(path, natoms):
            yield frame
    if number == 0:
        raise RuntimeError("no {}_*.lammpstrj files in {}".format(prefix, config_dir))


def analyse_star(config_dir, params):
    """Process the trajectories of a star run.

    Returns:
        ``(histograms, tables)``: dict name -> :class:`Histogram`, and a dict
        name -> ``(header, rows)`` (``loading``: colloids within rs and 2 rs per
        frame; empty if the run has no colloid).

    Raises:
        RuntimeError: if no frame could be read.
    """
    lbox = params.box[0]
    rs = star_radius(params.arms, params.ns)
    colloid = bool(params.colloids)
    natoms = params.arms * params.ns + 1 + (1 if colloid else 0)
    nbins_coll = int(round(lbox * 2))

    hist = {
        "mon_distr_anch": Histogram(3 * rs, 150),
        "mon_distr_cm": Histogram(3 * rs, 150),
    }
    if colloid:
        hist["coll_distr_anch"] = Histogram(lbox, nbins_coll)
        hist["coll_distr_anch_2rg"] = Histogram(lbox, nbins_coll)
        hist["coll_distr_cm"] = Histogram(lbox, nbins_coll)
    rgs = []
    loading = []

    for frame in trajectory_frames(config_dir, "star", natoms):
        obs = frame_observables(frame, lbox)
        hist["mon_distr_anch"].add(obs["mon_anch"])
        hist["mon_distr_cm"].add(obs["mon_cm"])
        rgs.append(obs["rg"])
        if colloid:
            hist["coll_distr_cm"].add(obs["coll_cm"])
            hist["coll_distr_anch"].add(obs["coll_anch"])
            hist["coll_distr_anch_2rg"].add(obs["coll_anch"][obs["coll_anch"] <= 2 * rs])
            loading.append((int(np.sum(obs["coll_anch"] <= rs)),
                            int(np.sum(obs["coll_anch"] <= 2 * rs))))

    if not rgs:
        raise RuntimeError("no frames found in {}".format(config_dir))

    hist["rg"] = Histogram(2 * rs, 200)
    hist["rg"].add(rgs)
    tables = {}
    if colloid:
        tables["loading"] = ("within_rs within_2rs", loading)
    return hist, tables


def bbrush_observables(frame, lbox):
    """Compute the per-frame quantities of a bottlebrush.

    Args:
        frame: ``(natoms, 5)`` array ``id type x y z`` (types: 1 side chain,
            2 backbone, 3 and 4 colloids).
        lbox: edge of the cubic periodic box.

    Returns:
        dict with ``ree`` (backbone end-to-end distance), ``rg_backbone``,
        ``rg`` (whole brush, types 1 and 2) and the distance arrays
        ``mon_cm``, ``coll_a_cm``, ``coll_b_cm`` from the brush centre of mass.
    """
    types = frame[:, 1].astype(int)
    backbone = frame[types == TYPE_BB_BACKBONE]
    backbone = backbone[np.argsort(backbone[:, 0])][:, 2:5]

    # Follow the backbone bond by bond so that it is continuous across the box.
    steps = np.diff(backbone, axis=0)
    steps -= lbox * np.round(steps / lbox)
    chain = backbone[0] + np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    cm_backbone = chain.mean(axis=0)

    # Unwrap every atom to its periodic image closest to the backbone centre.
    delta = frame[:, 2:5] - cm_backbone
    delta -= lbox * np.round(delta / lbox)
    xyz = cm_backbone + delta

    brush = xyz[(types == TYPE_BB_SIDE) | (types == TYPE_BB_BACKBONE)]
    cm = brush.mean(axis=0)
    return {
        "ree": float(np.linalg.norm(chain[-1] - chain[0])),
        "rg_backbone": float(np.sqrt(np.mean(np.sum((chain - cm_backbone) ** 2, axis=1)))),
        "rg": float(np.sqrt(np.mean(np.sum((brush - cm) ** 2, axis=1)))),
        "mon_cm": np.linalg.norm(brush - cm, axis=1),
        "coll_a_cm": np.linalg.norm(xyz[types == TYPE_BB_COLL_A] - cm, axis=1),
        "coll_b_cm": np.linalg.norm(xyz[types == TYPE_BB_COLL_B] - cm, axis=1),
    }


def analyse_bbrush(config_dir, params):
    """Process the trajectories of a bottlebrush run (see :func:`analyse_star`)."""
    lbox = params.box[0]
    colloid = bool(params.colloids)
    hist = {"mon_distr_cm": Histogram(lbox / 2, 150)}
    if colloid:
        hist["coll_a_distr_cm"] = Histogram(lbox, int(round(lbox * 2)))
        hist["coll_b_distr_cm"] = Histogram(lbox, int(round(lbox * 2)))
    rgs = []
    rows = []

    for index, frame in enumerate(trajectory_frames(config_dir, "bbrush")):
        obs = bbrush_observables(frame, lbox)
        hist["mon_distr_cm"].add(obs["mon_cm"])
        if colloid:
            hist["coll_a_distr_cm"].add(obs["coll_a_cm"])
            hist["coll_b_distr_cm"].add(obs["coll_b_cm"])
        rgs.append(obs["rg"])
        rows.append((index, obs["ree"], obs["rg_backbone"], obs["rg"]))

    if not rgs:
        raise RuntimeError("no frames found in {}".format(config_dir))
    hist["rg"] = Histogram(lbox / 2, 200)
    hist["rg"].add(rgs)
    return hist, {"end_to_end": ("frame ree rg_backbone rg", rows)}


def analyse(config_dir, params):
    """Process all trajectory files of a run, whatever its architecture."""
    if params.architecture == "star":
        return analyse_star(config_dir, params)
    return analyse_bbrush(config_dir, params)


def write_results(hist, tables, out_dir="analysis"):
    """Write one ``<name>.txt`` per histogram and per table."""
    create_dir(out_dir)
    for name, h in hist.items():
        with open(os.path.join(out_dir, name + ".txt"), "w") as handle:
            handle.write("# x    y\n")
            for x, y in zip(h.x, h.y):
                handle.write("{0} {1}\n".format(x, y))
    for name, (header, rows) in tables.items():
        with open(os.path.join(out_dir, name + ".txt"), "w") as handle:
            handle.write("# {}\n".format(header))
            for row in rows:
                handle.write(" ".join(str(v) for v in row) + "\n")


def main():
    """Analyse the run in the current directory."""
    params = read_par()
    hist, tables = analyse("configurations", params)
    write_results(hist, tables)


if __name__ == "__main__":
    main()
