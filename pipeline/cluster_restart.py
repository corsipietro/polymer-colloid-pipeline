#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continue the runs of an earlier scan from their last restart file.

For every run directory below ``root`` it refreshes the scripts, sets
``restart 1`` in ``par.txt`` and starts the run again, in the ``mode`` stored in
its ``par.txt`` (override with ``--mode``). Output files of the continued run
get the next index (see ``functions.check_file_number``). Do not use it on runs
that are still running or queued.

Usage::

    python3 cluster_restart.py 2021_8_9_11_0_f60 [--ns 100] [--mode local] [--dry-run]
"""

import argparse
import os
import shutil

from functions import find_run_dirs, read_par, write_par
from launcher import RUN_FILES, start_simulation


def main():
    """Restart every run found below the given root directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--ns", type=int, default=None,
                        help="only runs with this number of monomers per arm")
    parser.add_argument("--mode", choices=["local", "cluster"], default=None,
                        help="override the mode stored in each par.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="update the files but do not start anything")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    runs = find_run_dirs(args.root, args.ns)
    for run in runs:
        params = read_par(os.path.join(run, "par.txt"))
        if args.mode:
            params.mode = args.mode
        for script in RUN_FILES:
            shutil.copy(os.path.join(here, script), run)
        params.restart = 1
        write_par(os.path.join(run, "par.txt"), params)
        if not args.dry_run:
            start_simulation(run, params.mode)


if __name__ == "__main__":
    main()
