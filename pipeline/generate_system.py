#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the LAMMPS data file (``star.dat`` or ``bbrush.dat``) of the current run.

Reads ``par.txt``, picks the builder from its ``architecture`` key and creates
the output folders (``gyration``, ``configurations``, ``analysis``).

Usage (from a run directory)::

    python3 generate_system.py [--seed N]
"""

import argparse

import numpy as np

from functions import create_dir, read_par, star_colloid_diameters
from generate_bottle_brush import build_bottle_brush
from generate_star import build_star


def main():
    """Generate the data file for the architecture named in ``par.txt``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: 'seed' in par.txt, else random)")
    args = parser.parse_args()

    for directory in ("gyration", "configurations", "analysis"):
        create_dir(directory)

    params = read_par()
    seed = args.seed if args.seed is not None else params.seed
    rng = np.random.RandomState(seed)
    if params.architecture == "star":
        diameters = star_colloid_diameters(params)
        colloids = [(atom_type, count, diameters[index - 1])
                    for index, atom_type, count in params.species()]
        text = build_star(params.arms, params.ns, params.sigma, params.box, rng,
                          params.rcore, colloids, n_types=2 + params.n_species)
    else:
        text = build_bottle_brush(params.nb, params.ns, params.sigma, params.step,
                                  params.box, rng, n_types=2 + params.n_species)
    with open(params.spec["data_file"], "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
