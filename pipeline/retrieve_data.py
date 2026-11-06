#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect one analysis file of every run of a scan into a single folder.

By default it gathers the colloid distance distribution: ``coll_distr_anch``
for stars, ``coll_a_distr_cm`` for bottlebrushes (use ``--name`` for another
histogram, e.g. ``rg``). Each file is copied to
``<out>/<name>_<label>.txt`` (header ``x y``), where the label lists the main
parameters of the run. Runs not analysed yet are reported and skipped.

Usage::

    python3 retrieve_data.py 2021_8_9_11_0_f60 [--out data] [--name rg]
"""

import argparse
import os

from functions import create_dir, find_run_dirs, read_par

DEFAULT_NAME = {"star": "coll_distr_anch", "bottlebrush": "coll_a_distr_cm"}


def run_label(params):
    """Return a file-name label for a run, e.g. ``100_50_1.8_0.2`` (ns, f, q, eps, q2)."""
    one, two = params.present
    if params.architecture == "star":
        parts = [params.ns, params.arms]
    else:
        parts = [params.ns, params.nb, params.step]
    if one or two:
        parts.append(params.eps)
    if one and params.q is not None:
        parts.append(params.q)
    if two and params.q2 is not None:
        parts.append(params.q2)
    return "_".join(str(p) for p in parts)


def main():
    """Gather the chosen analysis file of all runs into the output folder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="directory created by cluster_submit.py")
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    parser.add_argument("--name", default=None,
                        help="analysis file to collect, without .txt (default: per architecture)")
    args = parser.parse_args()

    create_dir(args.out)
    for run in find_run_dirs(args.root):
        params = read_par(os.path.join(run, "par.txt"))
        name = args.name or DEFAULT_NAME[params.architecture]
        source = os.path.join(run, "analysis", name + ".txt")
        if not os.path.isfile(source):
            print("skipped (no {}): {}".format(name, run))
            continue
        target = os.path.join(args.out, "{}_{}.txt".format(name, run_label(params)))
        with open(source) as src, open(target, "w") as dst:
            next(src)                      # drop the "# x    y" header
            dst.write("x y\n")
            dst.writelines(src)


if __name__ == "__main__":
    main()
