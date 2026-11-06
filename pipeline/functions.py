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
# Default number of colloids per species: one for the star, 40 for the bottlebrush.
DEFAULT_NCOLL = {"star": 1, "bottlebrush": 40}
# Which colloid species are present when par.txt does not say: the star has one colloid
# (species 1), the bottlebrush the two species of the original in.bbrush.
DEFAULT_PRESENT = {"star": (1, 0), "bottlebrush": (1, 1)}

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
        colloid1,     1/0: colloid species 1 (atom type 3) / species 2 (atom type 4) present or
        colloid2      not, independently; both 0 = the polymer alone. The polymer itself has
                      two particle species that are always there: core (star) or backbone
                      (bottlebrush) = type 2, and arm / side-chain monomers = type 1.
                      Defaults: star 1/0, bottlebrush 1/1. (``colloids=n`` is accepted as a
                      shorthand: n = 0, 1, 2 sets colloid1 = n >= 1, colloid2 = n >= 2.)
        ncoll1/ncoll2 number of colloids of species 1 / 2;
                      default 1 for the star, 40 for the bottlebrush
        eps, eps2     colloid-monomer attraction strength of species 1 / 2 (eps2 = eps if None)
        seed          optional integer seed (None = random)
    Star only:
        arms          number of arms f
        q, q2         size ratio polymer / colloid of species 1 / 2: star radius over
                      colloid radius, i.e. colloid diameter = 2 rs / q
        rcore         core radius (1.75 fits at most ~42 arms; f=50 needs >= 2.0)
    Bottlebrush only:
        nb, step      backbone length and grafting spacing (one side chain every ``step``)
        q, q2         size ratio effective-star radius / colloid radius of species 1 / 2
                      (see ``generate_bottle_brush.colloid_diameters``); if absent, the
                      explicit diameters dcoll1, dcoll2 (defaults 14 and 10) are used
    """

    _FIELDS = ("architecture", "mode", "ns", "sigma", "lbox", "restart", "colloid1", "colloid2",
               "eps", "eps2", "seed", "arms", "q", "q2", "rcore", "nb", "step",
               "ncoll1", "dcoll1", "ncoll2", "dcoll2")

    def __init__(self, **kwargs):
        """Set the defaults, then override them with ``kwargs``."""
        self.architecture = "star"
        self.mode = "cluster"
        self.ns = None
        self.sigma = 1.0
        self.lbox = None
        self.restart = 0
        self.colloid1 = self.colloid2 = None
        self.eps = None
        self.eps2 = None
        self.seed = None
        self.arms = None
        self.q = None
        self.q2 = None
        self.rcore = DEFAULT_RCORE
        self.nb = None
        self.step = None
        self.ncoll1 = self.ncoll2 = None
        self.dcoll1, self.dcoll2 = 14.0, 10.0
        for key, value in kwargs.items():
            if key == "colloids":                   # shorthand: number of species
                if value is not None:
                    self.colloid1, self.colloid2 = int(value >= 1), int(value >= 2)
            elif key not in self._FIELDS:
                raise TypeError("unknown parameter: {}".format(key))
            else:
                setattr(self, key, value)

    @property
    def box(self):
        """``[Lx, Ly, Lz]`` of the cubic box."""
        return [self.lbox] * 3

    @property
    def spec(self):
        """File names of the architecture (see :data:`ARCHITECTURES`)."""
        return ARCHITECTURES[self.architecture]

    @property
    def present(self):
        """``(colloid1, colloid2)`` as 0/1 flags, with the architecture defaults filled in."""
        default = DEFAULT_PRESENT[self.architecture]
        return (default[0] if self.colloid1 is None else self.colloid1,
                default[1] if self.colloid2 is None else self.colloid2)

    @property
    def n_species(self):
        """Highest colloid atom type index declared minus 2: 0, 1 or 2.

        Atom types must be contiguous, so asking for species 2 alone still declares
        type 3 (unused).
        """
        one, two = self.present
        return 2 if two else (1 if one else 0)

    @property
    def counts(self):
        """``(n1, n2)``: colloids of species 1 and 2 actually present in the system."""
        default = DEFAULT_NCOLL[self.architecture]
        n1 = default if self.ncoll1 is None else self.ncoll1
        n2 = default if self.ncoll2 is None else self.ncoll2
        one, two = self.present
        return (n1 if one else 0, n2 if two else 0)

    @property
    def n_colloids(self):
        """Total number of colloids in the system."""
        return sum(self.counts)

    def species(self):
        """Return ``[(index, atom_type, count), ...]`` for the species that are present."""
        return [(i, 2 + i, n) for i, n in enumerate(self.counts, 1) if n > 0]

    def replace(self, **changes):
        """Return a copy with some attributes changed."""
        new = Params(**dict((k, getattr(self, k)) for k in self._FIELDS))
        for key, value in changes.items():
            if key == "colloids":                   # shorthand: number of species
                new.colloid1, new.colloid2 = int(value >= 1), int(value >= 2)
            elif key not in self._FIELDS:
                raise TypeError("unknown parameter: {}".format(key))
            else:
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


def star_colloid_diameters(params):
    """Return the diameters ``(d1, d2)`` of the colloid species of a star run.

    The diameter is ``2 * rs / q`` (``q`` for species 1, ``q2`` for species 2),
    i.e. ``q`` is the size ratio star radius / colloid radius. A species that is not
    present has diameter None.
    """
    rs = star_radius(params.arms, params.ns)
    d1 = 2 * rs / params.q if params.q is not None else None
    d2 = 2 * rs / params.q2 if params.q2 is not None else None
    return d1, d2


def read_par(path="par.txt"):
    """Read a ``par.txt`` file and return a :class:`Params`.

    The file holds one ``key value`` pair per line. Keys:

    * all: ``architecture`` (default star), ``mode`` (default cluster),
      ``ns``, ``sigma``, ``lbox``, ``restart``, ``seed``, ``colloid1``, ``colloid2``
      (0/1, each colloid species present or not; default star 1/0, bottlebrush 1/1;
      the legacy ``colloids`` = 0/1/2 is also understood), ``ncoll1``, ``ncoll2``
      (colloids per species), ``eps``, ``eps2``;
    * star: ``f``; ``eps``; ``q`` if species 1 is present, ``q2`` if species 2 is; optional ``rcore``;
    * bottlebrush: ``nb``, ``step``; ``eps`` (default 0.1); ``q`` / ``q2`` (size ratios)
      or the explicit diameters ``dcoll1`` / ``dcoll2`` (defaults 14 and 10).

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

    default = DEFAULT_PRESENT[architecture]
    if "colloids" in values:                       # legacy shorthand: number of species
        n = int(values["colloids"])
        if n not in (0, 1, 2):
            raise ValueError("{}: colloids must be 0, 1 or 2, got {}".format(path, n))
        default = (int(n >= 1), int(n >= 2))
    flags = []
    for key, fallback in (("colloid1", default[0]), ("colloid2", default[1])):
        flag = int(values[key]) if key in values else fallback
        if flag not in (0, 1):
            raise ValueError("{}: {} must be 0 or 1, got {}".format(path, key, flag))
        flags.append(flag)
    required = _COMMON_REQUIRED + _REQUIRED[architecture]
    if architecture == "star":
        if flags[0] or flags[1]:
            required += ("eps",)
        if flags[0]:
            required += ("q",)
        if flags[1]:
            required += ("q2",)
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError("{} is missing the keys: {}".format(path, ", ".join(missing)))

    def optional_int(key):
        """Return ``int(values[key])`` or None if the key is absent."""
        return int(values[key]) if key in values else None

    params = Params(
        architecture=architecture, mode=mode, colloid1=flags[0], colloid2=flags[1],
        ns=int(values["ns"]), sigma=values["sigma"], lbox=values["lbox"],
        restart=int(values["restart"]), seed=optional_int("seed"),
        eps=values.get("eps"), eps2=values.get("eps2"),
        q=values.get("q"), q2=values.get("q2"),
        ncoll1=optional_int("ncoll1"), ncoll2=optional_int("ncoll2"),
    )
    if architecture == "star":
        params.arms = int(values["f"])
        params.rcore = values.get("rcore", DEFAULT_RCORE)
    else:
        params.nb = int(values["nb"])
        params.step = int(values["step"])
        if params.eps is None:
            params.eps = DEFAULT_BBRUSH_EPS
        params.dcoll1 = values.get("dcoll1", params.dcoll1)
        params.dcoll2 = values.get("dcoll2", params.dcoll2)
    return params


def write_par(path, params):
    """Write ``params`` to a ``par.txt`` file readable by :func:`read_par`."""
    one, two = params.present
    lines = [("architecture", params.architecture), ("mode", params.mode),
             ("colloid1", one), ("colloid2", two)]
    if params.architecture == "star":
        lines += [("f", params.arms), ("ns", params.ns), ("sigma", params.sigma),
                  ("lbox", params.lbox), ("rcore", params.rcore)]
    else:
        lines += [("nb", params.nb), ("ns", params.ns), ("sigma", params.sigma),
                  ("step", params.step), ("lbox", params.lbox)]
    if one or two:
        lines += [("eps", params.eps)]
    if one:
        if params.ncoll1 is not None:
            lines += [("ncoll1", params.ncoll1)]
        if params.q is not None:
            lines += [("q", params.q)]
        elif params.architecture == "bottlebrush":
            lines += [("dcoll1", params.dcoll1)]
    if two:
        if params.eps2 is not None:
            lines += [("eps2", params.eps2)]
        if params.ncoll2 is not None:
            lines += [("ncoll2", params.ncoll2)]
        if params.q2 is not None:
            lines += [("q2", params.q2)]
        elif params.architecture == "bottlebrush":
            lines += [("dcoll2", params.dcoll2)]
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
