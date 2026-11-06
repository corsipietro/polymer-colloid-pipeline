#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the trajectory analysis (``post_processing.py``) of every run of a scan.

In cluster mode one PBS job per run is submitted; in local mode the analyses
run one after the other on this machine.

Usage::

    python3 post_cluster_submit.py 2021_8_9_11_0_f60 [--ns 100] [--mode local] [--dry-run]
"""

import argparse
import os
import shutil

from functions import find_run_dirs, read_par
from launcher import POST_FILES, start_analysis


def main():
    """Copy the analysis scripts into each run and start them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--ns", type=int, default=None,
                        help="only runs with this number of monomers per arm")
    parser.add_argument("--mode", choices=["local", "cluster"], default=None,
                        help="override the mode stored in each par.txt")
    parser.add_argument("--dry-run", action="store_true",
                        help="copy the scripts but do not start anything")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for run in find_run_dirs(args.root, args.ns):
        for script in POST_FILES:
            shutil.copy(os.path.join(here, script), run)
        if not args.dry_run:
            start_analysis(run, args.mode or read_par(os.path.join(run, "par.txt")).mode)


if __name__ == "__main__":
    main()
