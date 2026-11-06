#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Submit the trajectory analysis (``post_processing.py``) of every run of a scan.

Usage::

    python3 post_cluster_submit.py 2021_8_9_11_0_f60 [--ns 100] [--dry-run]
"""

import argparse
import os
import shutil
import subprocess

from functions import find_run_dirs

FILES = ["post_processing.py", "functions.py", "post_py.csh"]


def main():
    """Copy the analysis scripts into each run and submit them."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--ns", type=int, default=None,
                        help="only runs with this number of monomers per arm")
    parser.add_argument("--dry-run", action="store_true",
                        help="copy the scripts but do not submit")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for run in find_run_dirs(args.root, args.ns):
        for script in FILES:
            shutil.copy(os.path.join(here, script), run)
        if not args.dry_run:
            subprocess.call(["./post_py.csh"], cwd=run)


if __name__ == "__main__":
    main()
