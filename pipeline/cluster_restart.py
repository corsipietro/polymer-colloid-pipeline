#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continue the runs of an earlier scan from their last restart file.

For every run directory below ``ROOT`` it refreshes the generation scripts,
sets ``restart 1`` in ``par.txt`` and resubmits ``start_py.csh``. Output files
of the continued run get the next index (see ``functions.check_file_number``).
Do not use it on runs that are still in the queue.

Usage::

    python3 cluster_restart.py 2021_8_9_11_0_f60 [--ns 100] [--dry-run]
"""

import argparse
import os
import shutil
import subprocess

from functions import find_run_dirs, read_par, write_par

FILES = ["generate_star.py", "generate_input.py", "functions.py", "start_py.csh"]


def main():
    """Restart every run found below the given root directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--ns", type=int, default=None,
                        help="only runs with this number of monomers per arm")
    parser.add_argument("--dry-run", action="store_true",
                        help="update the files but do not submit")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for run in find_run_dirs(args.root, args.ns):
        params = read_par(os.path.join(run, "par.txt"))
        for script in FILES:
            shutil.copy(os.path.join(here, script), run)
        write_par(os.path.join(run, "par.txt"), params.arms, params.ns, params.sigma,
                  params.box[0], params.q, params.eps, restart=1, rcore=params.rcore)
        if not args.dry_run:
            subprocess.call(["./start_py.csh"], cwd=run)


if __name__ == "__main__":
    main()
