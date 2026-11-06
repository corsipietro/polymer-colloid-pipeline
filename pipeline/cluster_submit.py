#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the run directories of a parameter scan and submit them to the queue.

For every combination of (arms, monomers per arm, eps, q) it creates
``<timestamp>_f<arms>/f_<arms>/ns_<ns>/eps_<eps>/q_<q>/``, copies the
generation scripts there, writes ``par.txt`` and launches ``start_py.csh``
(which submits a PBS job).

Edit the PARAMETERS block, then run from this directory::

    python3 cluster_submit.py            # create and submit
    python3 cluster_submit.py --dry-run  # only create the directories
"""

import argparse
import os
import shutil
import subprocess
from datetime import datetime

import numpy as np

from functions import create_dir, star_radius, write_par

############################
######## PARAMETERS ########
############################

ARMS = [50]                                   # arms per star (f)
NS = [100]                                    # monomers per arm
RATIO = [1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8]   # q = star radius / colloid radius
EPS = [0.05, 0.1, 0.15, 0.2, 0.25]            # colloid-monomer attraction strength
PHI = 0.001                                   # volume fraction -> sets the box size
SIGMA = 1                                     # monomer diameter
RCORE = 2.25                                  # core radius; 1.75 cannot fit f=50 (needs >= 2.0)

############################

FILES = ["generate_star.py", "generate_input.py", "functions.py", "start_py.csh"]


def box_length(arms, ns, q, phi=PHI):
    """Return the box edge giving volume fraction ``phi`` for one star + one colloid."""
    rs = star_radius(arms, ns)
    rc = rs / q
    volume = (4 / 3) * np.pi * (rc ** 3 + rs ** 3) / phi
    return volume ** (1 / 3)


def main():
    """Build the directory tree and submit every run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="create directories and par.txt but do not submit")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    now = datetime.now()
    tag = "{}_{}_{}_{}_{}_f{}".format(now.year, now.month, now.day, now.hour,
                                      now.minute, "-".join(str(a) for a in ARMS))
    root = os.path.join(os.getcwd(), tag)

    for arms in ARMS:
        for ns in NS:
            for eps in EPS:
                for q in RATIO:
                    run = os.path.join(root, "f_{}".format(arms), "ns_{}".format(ns),
                                       "eps_{}".format(eps), "q_{}".format(q))
                    create_dir(run)
                    for script in FILES:
                        shutil.copy(os.path.join(here, script), run)
                    write_par(os.path.join(run, "par.txt"), arms, ns, SIGMA,
                              box_length(arms, ns, q), q, eps, restart=0, rcore=RCORE)
                    if not args.dry_run:
                        subprocess.call(["./start_py.csh"], cwd=run)
    print("Runs in {}".format(root))


if __name__ == "__main__":
    main()
