#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the polymer + colloid simulation pipeline.

Two architectures are supported (key ``architecture`` in ``par.txt``):

* ``star``         a star polymer (f arms of ns monomers) with, optionally, one colloid;
* ``bottlebrush``  a bottlebrush (backbone of nb monomers, side chains of ns
                   monomers every ``step`` backbone monomers) with, optionally,
                   two species of colloids.

Every script in this directory is copied into each run directory, so this
module must stay self-contained (numpy only) and compatible with the Python 3.4
interpreter used on the cluster nodes (no f-strings, no type annotations).
"""

import os
import re

import numpy as np

# Per-architecture file names. ``prefix`` is used for the output files
# (configurations/<prefix>_<n>.lammpstrj) and for the restart file.
ARCHITECTURES = {
    "star": {"prefix": "star", "data_file": "star.dat",
             "input_file": "in.star", "restart_file": "star2.restart"},
    "bottlebrush": {"prefix": "bbrush", "data_file": "bbrush.dat",
                    "input_file": "in.bbrush", "restart_file": "bbrush2.restart"},
}

# "local": run on this machine; "cluster": submit to the PBS queue.
MODES = ("local", "cluster")

DEFAULT_RCORE = 1.75          # star core radius
DEFAULT_BBRUSH_EPS = 0.1      # bottlebrush: colloid-monomer attraction strength

_STRING_KEYS = ("architecture", "mode")
_COMMON_REQUIRED = ("ns", "sigma", "lbox", "restart")
_REQUIRED = {
    "star": ("f",),
    "bottlebrush": ("nb", "step"),
}


class Params(object):
    """Parameters of one run, as stored in ``par.txt``.

    Common attributes:
        architecture  ``"star"`` or ``"bottlebrush"``
        mode          ``"local"`` or ``"cluster"``
        ns            monomers per arm (star) / per side chain (bottlebrush)
        sigma         monomer diameter (LJ units)
        lbox          edge of the cubic periodic box (``box`` = ``[lbox] * 3``)
        restart       1 to continue a previous run from its restart file
        colloids      1 if colloids are present, 0 for the bare polymer
        eps           colloid-monomer attraction strength
        seed          optional integer seed (None = random)
    Star only:
        arms          number of arms f
        q             star radius / colloid radius
        rcore         core radius (1.75 fits at most ~42 arms; f=50 needs >= 2.0)
    Bottlebrush only:
        nb, step      backbone length and grafting spacing (one side chain every ``step``)
        ncoll1, dcoll1, ncoll2, dcoll2   number and diameter of the two colloid species
    """

    _FIELDS = ("architecture", "mode", "ns", "sigma", "lbox", "restart", "colloids",
               "eps", "seed", "arms", "q", "rcore", "nb", "step",
               "ncoll1", "dcoll1", "ncoll2", "dcoll2")

    def __init__(self, **kwargs):
        """Set the defaults, then override them with ``kwargs``."""
        self.architecture = "star"
        self.mode = "cluster"
        self.ns = None
        self.sigma = 1.0
        self.lbox = None
        self.restart = 0
        self.colloids = 1
        self.eps = None
        self.seed = None
        self.arms = None
        self.q = None
        self.rcore = DEFAULT_RCORE
        self.nb = None
        self.step = None
        self.ncoll1, self.dcoll1 = 40, 14.0
        self.ncoll2, self.dcoll2 = 40, 10.0
        for key, value in kwargs.items():
            if key not in self._FIELDS:
                raise TypeError("unknown parameter: {}".format(key))
            setattr(self, key, value)

    @property
    def box(self):
        """``[Lx, Ly, Lz]`` of the cubic box."""
        return [self.lbox] * 3

    @property
    def spec(self):
        """File names of the architecture (see :data:`ARCHITECTURES`)."""
        return ARCHITECTURES[self.architecture]

    def replace(self, **changes):
        """Return a copy with some attributes changed."""
        new = Params(**dict((k, getattr(self, k)) for k in self._FIELDS))
        for key, value in changes.items():
            setattr(new, key, value)
        return new

    def __eq__(self, other):
        """Two Params are equal if all their fields are."""
        return isinstance(other, Params) and self.__dict__ == other.__dict__

    def __repr__(self):
        """Readable representation."""
        return "Params({})".format(", ".join(
            "{}={!r}".format(k, getattr(self, k)) for k in self._FIELDS))


def create_dir(path):
    """Create ``path`` (and parents) if it does not exist yet."""
    os.makedirs(path, exist_ok=True)


def star_radius(arms, ns):
    """Return the Flory-like radius of a star: ``0.6 * f**(1/5) * ns**0.588``."""
    return 0.6 * arms ** (1 / 5) * ns ** 0.588


def read_par(path="par.txt"):
    """Read a ``par.txt`` file and return a :class:`Params`.

    The file holds one ``key value`` pair per line. Keys:

    * all: ``architecture`` (default star), ``mode`` (default cluster),
      ``ns``, ``sigma``, ``lbox``, ``restart``, ``colloids`` (default 1), ``seed``;
    * star: ``f``, and, if colloids are present, ``q`` and ``eps``; optional ``rcore``;
    * bottlebrush: ``nb``, ``step``; optional ``eps`` (default 0.1), ``ncoll1``,
      ``dcoll1``, ``ncoll2``, ``dcoll2``.

    Files written before the ``architecture`` key existed are read as star runs.

    Raises:
        KeyError: if a required key is missing.
        ValueError: for an unknown architecture or mode.
    """
    values = {}
    with open(path) as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) == 2:
                key, value = tokens
                values[key] = value if key in _STRING_KEYS else float(value)

    architecture = values.get("architecture", "star")
    mode = values.get("mode", "cluster")
    if architecture not in ARCHITECTURES:
        raise ValueError("{}: unknown architecture '{}' (use one of: {})".format(
            path, architecture, ", ".join(sorted(ARCHITECTURES))))
    if mode not in MODES:
        raise ValueError("{}: unknown mode '{}' (use one of: {})".format(
            path, mode, ", ".join(MODES)))

    colloids = int(values.get("colloids", 1))
    required = _COMMON_REQUIRED + _REQUIRED[architecture]
    if architecture == "star" and colloids:
        required += ("q", "eps")
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError("{} is missing the keys: {}".format(path, ", ".join(missing)))

    def optional_int(key):
        """Return ``int(values[key])`` or None if the key is absent."""
        return int(values[key]) if key in values else None

    params = Params(
        architecture=architecture, mode=mode, colloids=colloids,
        ns=int(values["ns"]), sigma=values["sigma"], lbox=values["lbox"],
        restart=int(values["restart"]), seed=optional_int("seed"),
        eps=values.get("eps"),
    )
    if architecture == "star":
        params.arms = int(values["f"])
        params.q = values.get("q")
        params.rcore = values.get("rcore", DEFAULT_RCORE)
    else:
        params.nb = int(values["nb"])
        params.step = int(values["step"])
        if params.eps is None:
            params.eps = DEFAULT_BBRUSH_EPS
        params.ncoll1 = int(values.get("ncoll1", params.ncoll1))
        params.dcoll1 = values.get("dcoll1", params.dcoll1)
        params.ncoll2 = int(values.get("ncoll2", params.ncoll2))
        params.dcoll2 = values.get("dcoll2", params.dcoll2)
    return params


def write_par(path, params):
    """Write ``params`` to a ``par.txt`` file readable by :func:`read_par`."""
    lines = [("architecture", params.architecture), ("mode", params.mode),
             ("colloids", params.colloids)]
    if params.architecture == "star":
        lines += [("f", params.arms), ("ns", params.ns), ("sigma", params.sigma),
                  ("lbox", params.lbox)]
        if params.colloids:
            lines += [("q", params.q), ("eps", params.eps)]
        lines += [("rcore", params.rcore)]
    else:
        lines += [("nb", params.nb), ("ns", params.ns), ("sigma", params.sigma),
                  ("step", params.step), ("lbox", params.lbox)]
        if params.colloids:
            lines += [("eps", params.eps), ("ncoll1", params.ncoll1),
                      ("dcoll1", params.dcoll1), ("ncoll2", params.ncoll2),
                      ("dcoll2", params.dcoll2)]
    lines += [("restart", params.restart)]
    if params.seed is not None:
        lines += [("seed", params.seed)]
    with open(path, "w") as handle:
        for key, value in lines:
            handle.write("{} {}\n".format(key, value))


def _indices_in(directory, pattern):
    """Return the integers captured by ``pattern`` in the file names of ``directory``."""
    if not os.path.isdir(directory):
        return []
    found = []
    for name in os.listdir(directory):
        match = re.match(pattern, name)
        if match:
            found.append(int(match.group(1)))
    return found


def check_file_number(restart, architecture="star", gyration_dir="gyration",
                      config_dir="configurations"):
    """Return the index of the next output file of a (re)started run.

    A fresh run (``restart == 0`` or no restart file) writes index 0. When
    continuing, the index is one more than the highest index already used by
    ``gyration/gyr_all_<n>.txt`` or ``configurations/<prefix>_<n>.lammpstrj``,
    so that earlier output is never overwritten (also past 9 restarts).
    """
    spec = ARCHITECTURES[architecture]
    if restart != 1 or not os.path.isfile(spec["restart_file"]):
        return 0
    used = (_indices_in(gyration_dir, r"gyr_all_([0-9]+)\.txt$")
            + _indices_in(config_dir, spec["prefix"] + r"_([0-9]+)\.lammpstrj$"))
    return max(used) + 1 if used else 0


def find_run_dirs(root, ns=None):
    """Return the sorted run directories below ``root``.

    A run directory is any directory that contains a ``par.txt``.

    Args:
        root: top-level directory created by ``cluster_submit.py``.
        ns: if given, keep only the runs with this number of monomers per arm.
    """
    runs = []
    for dir_name, _, file_names in os.walk(root):
        if "par.txt" not in file_names:
            continue
        if ns is not None and read_par(os.path.join(dir_name, "par.txt")).ns != ns:
            continue
        runs.append(dir_name)
    return sorted(runs)


def distance(a, b):
    """Return the Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
