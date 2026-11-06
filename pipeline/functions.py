#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the star-polymer + colloid simulation pipeline.

Every script in this directory is copied into each run directory on the
cluster, so this module must stay self-contained (numpy only) and compatible
with the Python 3.4 interpreter used on the cluster nodes (no f-strings, no
type annotations).
"""

import os
import re
from collections import namedtuple

import numpy as np

# Parameters of one run, as stored in ``par.txt``.
#   arms    number of arms (functionality f) of the star
#   ns      number of monomers per arm
#   sigma   monomer diameter (LJ units)
#   q       size ratio  star radius / colloid radius
#   eps     colloid-monomer interaction strength (LJ units)
#   box     [Lx, Ly, Lz] of the cubic periodic box
#   restart 1 to continue a previous run from ``star2.restart``, 0 otherwise
#   rcore   radius of the star core (optional key ``rcore``, default 1.75).
#           Large stars need a larger core: with 1.75 at most ~42 arms fit.
Params = namedtuple("Params", ["arms", "ns", "sigma", "q", "eps", "box", "restart", "rcore"])

DEFAULT_RCORE = 1.75

_REQUIRED_KEYS = ("f", "ns", "sigma", "lbox", "q", "eps", "restart")

# Matches the innermost run directory, e.g. ".../f_50/ns_100/eps_0.2/q_1.8".
RUN_DIR_PATTERN = re.compile(r"q_[0-9]+\.[0-9]+$")


def create_dir(path):
    """Create ``path`` (and parents) if it does not exist yet."""
    os.makedirs(path, exist_ok=True)


def star_radius(arms, ns):
    """Return the Flory-like radius of a star: ``0.6 * f**(1/5) * ns**0.588``."""
    return 0.6 * arms ** (1 / 5) * ns ** 0.588


def read_par(path="par.txt"):
    """Read a ``par.txt`` file and return a :class:`Params` tuple.

    The file holds one ``key value`` pair per line (keys: f, ns, sigma, lbox,
    q, eps, restart and optionally rcore). Blank lines are ignored.

    Raises:
        KeyError: if a required key is missing.
    """
    values = {}
    with open(path) as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) == 2:
                values[tokens[0]] = float(tokens[1])

    missing = [key for key in _REQUIRED_KEYS if key not in values]
    if missing:
        raise KeyError("{} is missing the keys: {}".format(path, ", ".join(missing)))

    lbox = values["lbox"]
    return Params(
        arms=int(values["f"]),
        ns=int(values["ns"]),
        sigma=values["sigma"],
        q=values["q"],
        eps=values["eps"],
        box=[lbox, lbox, lbox],
        restart=int(values["restart"]),
        rcore=values.get("rcore", DEFAULT_RCORE),
    )


def write_par(path, arms, ns, sigma, lbox, q, eps, restart, rcore=DEFAULT_RCORE):
    """Write a ``par.txt`` file readable by :func:`read_par`."""
    with open(path, "w") as handle:
        handle.write("f {}\n".format(arms))
        handle.write("ns {}\n".format(ns))
        handle.write("sigma {}\n".format(sigma))
        handle.write("lbox {}\n".format(lbox))
        handle.write("q {}\n".format(q))
        handle.write("eps {}\n".format(eps))
        handle.write("restart {}\n".format(restart))
        handle.write("rcore {}\n".format(rcore))


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


def check_file_number(restart, restart_file="star2.restart",
                      gyration_dir="gyration", config_dir="configurations"):
    """Return the index of the next output file of a (re)started run.

    A fresh run (``restart == 0`` or no restart file) writes index 0. When
    continuing, the index is one more than the highest index already used by
    ``gyration/gyr_all_<n>.txt`` or ``configurations/star_<n>.lammpstrj``, so
    that earlier output is never overwritten (also past 9 restarts).
    """
    if restart != 1 or not os.path.isfile(restart_file):
        return 0
    used = (_indices_in(gyration_dir, r"gyr_all_([0-9]+)\.txt$")
            + _indices_in(config_dir, r"star_([0-9]+)\.lammpstrj$"))
    return max(used) + 1 if used else 0


def find_run_dirs(root, ns=None):
    """Return the sorted run directories (``.../q_<ratio>``) below ``root``.

    Args:
        root: top-level directory created by ``cluster_submit.py``.
        ns: if given, keep only the runs whose ``ns_<ns>`` path component
            matches exactly.
    """
    runs = []
    for dir_name, _, _ in os.walk(root):
        if not RUN_DIR_PATTERN.search(dir_name):
            continue
        if ns is not None and "ns_{}".format(ns) not in dir_name.split(os.sep):
            continue
        runs.append(dir_name)
    return sorted(runs)


def distance(a, b):
    """Return the Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
