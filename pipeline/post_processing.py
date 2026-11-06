#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse the trajectories of one run and write histograms to ``analysis/``.

Reads ``configurations/<prefix>_<n>.lammpstrj`` (prefix ``star`` or ``bbrush``,
n = 0, 1, ... until a file is missing) and ``par.txt``. The first frame of every
file is skipped: it is the configuration at the start of the (re)started run,
not an equilibrated one.

STAR output files (two columns ``x y``, bin centre and normalised histogram;
the colloid files are written only for the colloid species present; species 2
uses the suffix ``_b``, e.g. ``coll_distr_anch_b``):
  mon_distr_anch        monomer distance from the core
  mon_distr_cm          monomer distance from the centre of mass of the monomers
  coll_distr_anch       colloid distance from the core
  coll_distr_anch_2rg   same, only for frames where the colloid is within 2 rs
  coll_distr_cm         colloid distance from the centre of mass of the monomers
  rg                    distribution of the radius of gyration of the star
  loading.txt           per frame: number of colloids within rs and within 2 rs

``rs`` is the star radius ``0.6 * f**(1/5) * ns**0.588``. Distances use the
minimum image convention relative to the core.

BOTTLEBRUSH output files (the analysis of the original scripts; colloids optional):
  density_profile.txt   side-chain atoms vs distance from their anchor, orthogonal
                        and parallel to the local backbone direction (counts per frame)
  coll_dist_nnorm.txt   histogram of the colloid distance from the backbone (cylinder),
                        from the closest "effective star" and from the brush centre of mass
  coll_dist.txt         same, normalised (cylindrical / spherical shells)
  coll_dist_b*.txt      same for the second colloid species (type 4)
  end_to_end.txt        per frame: backbone end-to-end distance, backbone Rg, brush Rg
An "effective star" is a group of n consecutive backbone atoms with their side
chains, n = round(0.5 ns (sigma ns)^(3/5) / (1 + sigma ns)). Positions are unwrapped
around the backbone centre of mass, which assumes that the brush is smaller than
half the box.
"""

import os

import numpy as np

from functions import create_dir, read_par, star_radius
from generate_bottle_brush import set_anchors

# Lines of the header that precedes the atoms of every LAMMPS dump frame.
HEADER_LINES = 9

TYPE_MONOMER, TYPE_CORE, TYPE_COLLOID, TYPE_COLLOID_B = 1, 2, 3, 4          # star
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
        dict with arrays ``mon_anch``, ``mon_cm`` (monomer distances), the float
        ``rg`` and, for each colloid species present (``"a"`` = type 3, ``"b"`` =
        type 4), a dict with the arrays ``anch`` and ``cm`` (colloid distances from the
        core and from the centre of mass of the monomers).
    """
    types = frame[:, 1].astype(int)
    xyz = frame[:, 2:5]
    anchor = xyz[types == TYPE_CORE][0]

    # Unwrap every atom to its periodic image closest to the core.
    delta = xyz - anchor
    delta -= lbox * np.round(delta / lbox)
    xyz = anchor + delta

    monomers = xyz[types == TYPE_MONOMER]

    # The centre of mass uses the monomers only (core excluded).
    cm = monomers.mean(axis=0)

    # Radius of gyration of the whole star (core + monomers), the same
    # definition as LAMMPS ``compute gyration`` on group "type 1 2".
    star = xyz[(types == TYPE_MONOMER) | (types == TYPE_CORE)]
    rg = np.sqrt(np.mean(np.sum((star - star.mean(axis=0)) ** 2, axis=1)))

    result = {
        "mon_anch": np.linalg.norm(monomers - anchor, axis=1),
        "mon_cm": np.linalg.norm(monomers - cm, axis=1),
        "rg": rg,
    }
    for species, atom_type in (("a", TYPE_COLLOID), ("b", TYPE_COLLOID_B)):
        colloids = xyz[types == atom_type]
        if len(colloids):
            result[species] = {"anch": np.linalg.norm(colloids - anchor, axis=1),
                               "cm": np.linalg.norm(colloids - cm, axis=1)}
    return result


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
        name -> ``(header, rows)``. For each colloid species present there are the
        histograms ``coll_distr_anch``, ``coll_distr_anch_2rg``, ``coll_distr_cm`` and the
        table ``loading`` (colloids of the species within rs and 2 rs, per frame);
        species 2 uses the suffix ``_b``.

    Raises:
        RuntimeError: if no frame could be read.
    """
    lbox = params.box[0]
    rs = star_radius(params.arms, params.ns)
    natoms = params.arms * params.ns + 1 + params.n_colloids
    nbins_coll = int(round(lbox * 2))
    species = [(key, suffix) for key, suffix, n in
               zip(("a", "b"), ("", "_b"), params.counts) if n > 0]

    hist = {
        "mon_distr_anch": Histogram(3 * rs, 150),
        "mon_distr_cm": Histogram(3 * rs, 150),
    }
    for _, suffix in species:
        hist["coll_distr_anch" + suffix] = Histogram(lbox, nbins_coll)
        hist["coll_distr_anch_2rg" + suffix] = Histogram(lbox, nbins_coll)
        hist["coll_distr_cm" + suffix] = Histogram(lbox, nbins_coll)
    rgs = []
    loading = dict((suffix, []) for _, suffix in species)

    for frame in trajectory_frames(config_dir, "star", natoms):
        obs = frame_observables(frame, lbox)
        hist["mon_distr_anch"].add(obs["mon_anch"])
        hist["mon_distr_cm"].add(obs["mon_cm"])
        rgs.append(obs["rg"])
        for key, suffix in species:
            anch = obs[key]["anch"]
            hist["coll_distr_cm" + suffix].add(obs[key]["cm"])
            hist["coll_distr_anch" + suffix].add(anch)
            hist["coll_distr_anch_2rg" + suffix].add(anch[anch <= 2 * rs])
            loading[suffix].append((int(np.sum(anch <= rs)), int(np.sum(anch <= 2 * rs))))

    if not rgs:
        raise RuntimeError("no frames found in {}".format(config_dir))

    hist["rg"] = Histogram(2 * rs, 200)
    hist["rg"].add(rgs)
    tables = dict(("loading" + suffix, ("within_rs within_2rs", rows))
                  for suffix, rows in loading.items())
    return hist, tables


class BrushLayout(object):
    """Atom-id bookkeeping of a bottlebrush, derived from ``par.txt``.

    Attributes:
        nb: number of backbone atoms (ids 1..nb).
        anchors: backbone ids carrying a side chain.
        side: dict anchor -> list of the ids of its side-chain atoms.
        groups: list of ``(ids)`` arrays, one per "effective star": the backbone
            atoms ``k*n .. k*n+n-1`` (n = :attr:`blob_size`) with their side chains.
    """

    def __init__(self, params):
        """Compute anchors, side-chain ids and effective stars."""
        anchors, nb = set_anchors(params.step, params.nb)
        ns, sigma = params.ns, params.sigma
        self.nb = nb
        self.anchors = anchors
        self.side = {}
        for index, anchor in enumerate(anchors):
            first = nb + 1 + index * ns
            self.side[anchor] = list(range(first, first + ns))
        # Number of backbone monomers in an effective star.
        self.blob_size = max(1, int(round(0.5 * ns * (sigma * ns) ** (3 / 5) / (1 + sigma * ns))))
        self.groups = []
        for k in range(int(round(nb / self.blob_size)) + 1):
            ids = []
            for i in range(k * self.blob_size, k * self.blob_size + self.blob_size):
                if 1 <= i <= nb:
                    ids.append(i)
                    ids.extend(self.side.get(i, []))
            if ids:
                self.groups.append(np.array(ids))


def bbrush_frame(frame, lbox, layout):
    """Compute the per-frame quantities of a bottlebrush.

    Atoms are unwrapped to their periodic image closest to the backbone centre
    of mass (the backbone is first made continuous bond by bond), which assumes
    that the brush is smaller than half the box.

    Args:
        frame: ``(natoms, 5)`` array ``id type x y z`` (types: 1 side chain,
            2 backbone, 3 and 4 colloid species).
        lbox: edge of the cubic periodic box.
        layout: :class:`BrushLayout`.

    Returns:
        dict with ``ree`` (backbone end-to-end distance), ``rg_backbone``, ``rg``
        (brush, types 1 and 2); ``par`` and ``ort`` (distance of every side-chain
        atom from its anchor, parallel and orthogonal to the local backbone
        direction); and, for each colloid species present (``"a"`` = type 3,
        ``"b"`` = type 4), the distance arrays ``cyl`` (closest backbone atom),
        ``eff`` (closest effective-star centre) and ``cm`` (brush centre of mass).
    """
    frame = frame[np.argsort(frame[:, 0])]
    types = frame[:, 1].astype(int)
    xyz = frame[:, 2:5]
    nb = layout.nb

    steps = np.diff(xyz[:nb], axis=0)
    steps -= lbox * np.round(steps / lbox)
    chain = xyz[0] + np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    cm_backbone = chain.mean(axis=0)
    delta = xyz - cm_backbone
    delta -= lbox * np.round(delta / lbox)
    xyz = cm_backbone + delta

    backbone = xyz[:nb]
    brush = xyz[(types == TYPE_BB_SIDE) | (types == TYPE_BB_BACKBONE)]
    cm = brush.mean(axis=0)

    parallel, orthogonal = [], []
    for anchor in layout.anchors:
        i = anchor - 1
        bone = xyz[i + 1] - xyz[i] if anchor != nb else xyz[i] - xyz[i - 1]
        arm = xyz[np.array(layout.side[anchor]) - 1] - xyz[i]
        proj = arm.dot(bone) / bone.dot(bone)
        parallel.extend(np.abs(proj) * np.linalg.norm(bone))
        orthogonal.extend(np.linalg.norm(arm - np.outer(proj, bone), axis=1))

    result = {
        "ree": float(np.linalg.norm(backbone[-1] - backbone[0])),
        "rg_backbone": float(np.sqrt(np.mean(np.sum((backbone - backbone.mean(axis=0)) ** 2, axis=1)))),
        "rg": float(np.sqrt(np.mean(np.sum((brush - cm) ** 2, axis=1)))),
        "par": np.array(parallel),
        "ort": np.array(orthogonal),
    }
    centres = np.array([xyz[ids - 1].mean(axis=0) for ids in layout.groups])
    for species, atom_type in (("a", TYPE_BB_COLL_A), ("b", TYPE_BB_COLL_B)):
        colloids = xyz[types == atom_type]
        if len(colloids):
            result[species] = {
                "cyl": np.linalg.norm(colloids[:, None, :] - backbone[None, :, :], axis=2).min(axis=1),
                "eff": np.linalg.norm(colloids[:, None, :] - centres[None, :, :], axis=2).min(axis=1),
                "cm": np.linalg.norm(colloids - cm, axis=1),
            }
    return result


def analyse_bbrush(config_dir, params):
    """Process the trajectories of a bottlebrush run (see :func:`analyse_star`).

    Writes (as tables): ``density_profile`` (side-chain atoms vs distance
    parallel/orthogonal to the backbone, counts per frame), ``end_to_end`` and,
    for each colloid species present, the colloid distance distributions
    ``coll_dist_nnorm`` / ``coll_dist`` (species a, type 3) and
    ``coll_dist_nnorm_b`` / ``coll_dist_b`` (species b, type 4).
    """
    lbox = params.box[0]
    layout = BrushLayout(params)
    nb = layout.nb
    bins = np.linspace(0, nb, 2 * nb + 1)
    width = bins[1] - bins[0]
    x = 0.5 * (bins[:-1] + bins[1:])

    hist_ort = np.zeros(2 * nb)
    hist_par = np.zeros(2 * nb)
    dists = {"a": {"cyl": [], "eff": [], "cm": []}, "b": {"cyl": [], "eff": [], "cm": []}}
    n_coll = {"a": 0, "b": 0}
    rows = []
    counter = 0

    for frame in trajectory_frames(config_dir, "bbrush"):
        obs = bbrush_frame(frame, lbox, layout)
        hist_ort += np.histogram(obs["ort"], bins=bins)[0]
        hist_par += np.histogram(obs["par"], bins=bins)[0]
        for species in ("a", "b"):
            if species in obs:
                n_coll[species] = len(obs[species]["cm"])
                for kind in ("cyl", "eff", "cm"):
                    dists[species][kind].extend(obs[species][kind])
        rows.append((counter, obs["ree"], obs["rg_backbone"], obs["rg"]))
        counter += 1

    if counter == 0:
        raise RuntimeError("no frames found in {}".format(config_dir))

    tables = {
        "density_profile": ("x  ort  par", list(zip(x, hist_ort / counter, hist_par / counter))),
        "end_to_end": ("frame ree rg_backbone rg", rows),
    }
    n_eff = len(layout.groups)
    for species, suffix in (("a", ""), ("b", "_b")):
        if not n_coll[species]:
            continue
        number = counter * n_coll[species]
        counts = dict((k, np.histogram(v, bins=bins)[0]) for k, v in dists[species].items())
        tables["coll_dist_nnorm" + suffix] = (
            "x    y_cyl   y_eff   y_cm    counter",
            [(xi, c, e, m, counter) for xi, c, e, m in
             zip(x, counts["cyl"], counts["eff"], counts["cm"])])
        # Cylindrical shell (2 pi r dr) for the distance to the backbone, spherical
        # shell (4 pi r^2 dr) for the distances to the effective stars and to the
        # brush centre of mass; the first two are also divided by the number of
        # effective stars. -log(g(r)) is the effective potential.
        y_cyl = counts["cyl"] / (n_eff * number * 2 * np.pi * width) / x
        y_eff = counts["eff"] / (4 * np.pi * width * n_eff * number) / x ** 2
        y_cm = counts["cm"] / (4 * np.pi * width * number) / x ** 2
        tables["coll_dist" + suffix] = (
            "x    y_cyl   y_eff   y_cm", list(zip(x, y_cyl, y_eff, y_cm)))
    return {}, tables


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
