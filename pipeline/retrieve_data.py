#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the colloid distance distribution of every run into one folder.

For each run below ``root`` it copies ``analysis/coll_distr_anch.txt`` to
``<out>/coll_<ns>_<f>_<q>_<eps>.txt`` (two columns ``x y``, header ``x y``).
Runs without that file (not analysed yet) are reported and skipped.

Usage::

    python3 retrieve_data.py 2021_8_9_11_0_f60 [--out data]
"""

import argparse
import os
import re

from functions import create_dir, find_run_dirs

_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def run_label(run):
    """Return ``(ns, f, q, eps)`` as strings, parsed from a run directory path."""
    q = re.search(r"/q_" + _NUMBER, run).group(1)
    eps = re.search(r"/eps_" + _NUMBER, run).group(1)
    arms = re.search(r"/f_([0-9]+)", run).group(1)
    ns = re.search(r"/ns_([0-9]+)", run).group(1)
    return ns, arms, q, eps


def main():
    """Gather ``coll_distr_anch.txt`` of all runs into the output folder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    args = parser.parse_args()

    create_dir(args.out)
    for run in find_run_dirs(args.root):
        source = os.path.join(run, "analysis", "coll_distr_anch.txt")
        if not os.path.isfile(source):
            print("skipped (no analysis): {}".format(run))
            continue
        target = os.path.join(args.out, "coll_{}_{}_{}_{}.txt".format(*run_label(run)))
        with open(source) as src, open(target, "w") as dst:
            next(src)                      # drop the "# x    y" header
            dst.write("x y\n")
            dst.writelines(src)


if __name__ == "__main__":
    main()
