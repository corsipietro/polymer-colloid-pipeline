#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the run directories of a parameter scan and start every run.

Edit the PARAMETERS block (architecture, scan values, mode), then run from this
directory::

    python3 cluster_submit.py                  # create and start (mode from the file)
    python3 cluster_submit.py --mode local     # run on this machine
    python3 cluster_submit.py --dry-run        # only create directories and par.txt

Star scan:        <tag>/f_<f>/ns_<ns>/[eps_<eps>/q_<q>/[q2_<q2>/]]
Bottlebrush scan: <tag>/nb_<nb>/ns_<ns>/step_<step>/[eps_<eps>/[q_<q>/[q2_<q2>/]]]
Each run directory gets its own ``par.txt`` and a copy of the scripts.

In local mode the simulations are started in the background; run one scan point
at a time (the script refuses more than one run unless --allow-many is given).
"""

import argparse
import itertools
import os
import shutil
from datetime import datetime

import numpy as np

from functions import Params, create_dir, star_radius, write_par
from launcher import RUN_FILES, start_simulation

############################
######## PARAMETERS ########
############################

ARCHITECTURE = "star"       # "star" or "bottlebrush"
MODE = "cluster"            # "cluster" (PBS queue) or "local" (this machine)
COLLOID1 = 1                # colloid species 1 present (1) or not (0)
COLLOID2 = 0                # colloid species 2 present (1) or not (0); both 0 = polymer alone
SIGMA = 1                   # monomer diameter
NS = [100]                  # monomers per arm / per side chain

# --- star ---
ARMS = [50]                                   # arms per star (f)
RATIO = [1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]   # q = star radius / colloid radius (species 1)
EPS_STAR = [0.05, 0.1, 0.15, 0.2, 0.25]       # colloid-monomer attraction strength
RATIO2 = [2.6]                                # q2, species 2 (used only if COLLOID2 = 1)
NCOLL_STAR = (1, 1)                           # colloids of species 1 and 2
PHI = 0.001                                   # volume fraction -> sets the box size
RCORE = 2.25                                  # core radius (1.75 cannot fit f=50; needs >= 2.0)

# --- bottlebrush ---
NB = [99]                   # backbone monomers
STEP = [3]                  # backbone monomers between side chains
EPS_BBRUSH = [0.1]          # colloid-monomer attraction strength
LBOX_BBRUSH = 120           # box edge (must exceed nb*sigma and 2*ns*sigma)
NCOLL = (40, 40)            # colloids of species 1 and 2 (0 = species absent)
DCOLL = (14.0, 10.0)        # their diameters, used when RATIO_BBRUSH / RATIO2_BBRUSH are empty
RATIO_BBRUSH = []           # size ratios q = effective-star radius / colloid radius of species 1;
                            # if not empty they replace DCOLL[0] (diameter = 2 rc / q)
RATIO2_BBRUSH = []          # same for species 2 (replaces DCOLL[1])

############################


def box_length(arms, ns, q, phi=PHI):
    """Return the star box edge for volume fraction ``phi`` (one star, one colloid of ratio q).

    ``q=None`` (polymer alone) uses the star volume only.
    """
    rs = star_radius(arms, ns)
    rc = rs / q if q else 0.0
    volume = (4 / 3) * np.pi * (rc ** 3 + rs ** 3) / phi
    return volume ** (1 / 3)


def scan_runs(mode):
    """Yield ``(relative_path, Params)`` for every point of the configured scan."""
    one, two = COLLOID1, COLLOID2
    common = dict(mode=mode, colloid1=one, colloid2=two, sigma=SIGMA, restart=0)
    any_colloid = bool(one or two)
    if ARCHITECTURE == "star":
        epss = EPS_STAR if any_colloid else [None]
        qs = RATIO if one else [None]
        q2s = RATIO2 if two else [None]
        for arms, ns, eps, q, q2 in itertools.product(ARMS, NS, epss, qs, q2s):
            parts = ["f_{}".format(arms), "ns_{}".format(ns)]
            if any_colloid:
                parts.append("eps_{}".format(eps))
            if one:
                parts.append("q_{}".format(q))
            if two:
                parts.append("q2_{}".format(q2))
            lbox = box_length(arms, ns, q if one else (q2 if two else None))
            yield os.path.join(*parts), Params(
                architecture="star", arms=arms, ns=ns, q=q, q2=q2, eps=eps,
                lbox=lbox, rcore=RCORE, ncoll1=NCOLL_STAR[0], ncoll2=NCOLL_STAR[1], **common)
    elif ARCHITECTURE == "bottlebrush":
        epss = EPS_BBRUSH if any_colloid else [None]
        qs = RATIO_BBRUSH if (one and RATIO_BBRUSH) else [None]
        q2s = RATIO2_BBRUSH if (two and RATIO2_BBRUSH) else [None]
        for nb, ns, step, eps, q, q2 in itertools.product(NB, NS, STEP, epss, qs, q2s):
            parts = ["nb_{}".format(nb), "ns_{}".format(ns), "step_{}".format(step)]
            if any_colloid:
                parts.append("eps_{}".format(eps))
            if q is not None:
                parts.append("q_{}".format(q))
            if q2 is not None:
                parts.append("q2_{}".format(q2))
            yield os.path.join(*parts), Params(
                architecture="bottlebrush", nb=nb, ns=ns, step=step, eps=eps, q=q, q2=q2,
                lbox=LBOX_BBRUSH, ncoll1=NCOLL[0], ncoll2=NCOLL[1],
                dcoll1=DCOLL[0], dcoll2=DCOLL[1], **common)
    else:
        raise ValueError("ARCHITECTURE must be 'star' or 'bottlebrush'")


def main():
    """Build the directory tree and start every run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["local", "cluster"], default=MODE,
                        help="where to run (default: MODE in this file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="create directories and par.txt but do not start anything")
    parser.add_argument("--allow-many", action="store_true",
                        help="in local mode, allow starting more than one simulation")
    args = parser.parse_args()

    runs = list(scan_runs(args.mode))
    if args.mode == "local" and len(runs) > 1 and not (args.allow_many or args.dry_run):
        raise SystemExit("local mode: the scan has {} runs; running several simulations "
                         "on one machine is not recommended. Reduce the scan or pass "
                         "--allow-many.".format(len(runs)))

    here = os.path.dirname(os.path.abspath(__file__))
    now = datetime.now()
    size = ARMS if ARCHITECTURE == "star" else NB
    tag = "{}_{}_{}_{}_{}_{}{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                        "f" if ARCHITECTURE == "star" else "nb",
                                        "-".join(str(a) for a in size))
    root = os.path.join(os.getcwd(), tag)

    for relative, params in runs:
        run = os.path.join(root, relative)
        create_dir(run)
        for script in RUN_FILES:
            shutil.copy(os.path.join(here, script), run)
        write_par(os.path.join(run, "par.txt"), params)
        if not args.dry_run:
            start_simulation(run, args.mode)
    print("Runs in {}".format(root))


if __name__ == "__main__":
    main()
