#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the LAMMPS input script (``in.star`` or ``in.bbrush``) of the current run.

Common interactions (LJ units, reduced temperature 1.0, NVT):
  * polymer beads: purely repulsive (WCA-like, cut at the LJ minimum);
  * colloid-bead: attractive LJ of depth ``eps``, cut at 2.5 sigma_ij, with
    Lorentz mixing of the diameters;
  * colloid-colloid: repulsive;
  * FENE bonds (k = 30 / s^2, R0 = 1.5 s).

Star: one colloid of diameter ``2 rs / q`` (``rs`` = star radius, ``q`` = size
ratio), read from ``star.dat``; repulsion via shifted-force LJ.
Bottlebrush: ``ncoll1`` colloids of diameter ``dcoll1`` and ``ncoll2`` of
diameter ``dcoll2``, inserted at random positions with ``create_atoms``.

A fresh run reads the data file; with ``restart 1`` and an existing restart
file the run is continued, writing to the next output file index.

Usage (from a run directory containing ``par.txt``)::

    python3 generate_input.py
"""

import os
import random

import numpy as np

from functions import check_file_number, read_par, star_radius
from generate_bottle_brush import brush_size

# LJ energy scale of the FENE bonds and thermostat temperature.
ENERGY = 1
TEMPERATURE = "1.0"
CUT = 2 ** (1 / 6)  # minimum of the LJ potential, in units of sigma_ij


def fene_k(x):
    """FENE spring constant for a bond of length scale ``x``."""
    return 30 * ENERGY / (x ** 2)


def fene_r0(x):
    """FENE maximum extension for a bond of length scale ``x``."""
    return 1.5 * x


def build_input_star(params, file_number, continue_run):
    """Return the text of ``in.star``.

    Args:
        params: :class:`functions.Params` of a star run.
        file_number: index used in the names of the output files
            (``gyration/gyr_all_<n>.txt``, ``configurations/star_<n>.lammpstrj``).
        continue_run: True to read ``star2.restart`` instead of ``star.dat``.
    """
    arms, ns, sigma = params.arms, params.ns, params.sigma
    colloid = bool(params.colloids)
    r_core = params.rcore
    ntot = 1 + ns * arms

    s_core = r_core + sigma / 2

    out = []
    out.append("######################\n#   Initialization   #\n######################\n")
    out.append("\n")
    out.append("units lj\n")
    out.append("atom_style bond\n")
    out.append("special_bonds fene\n")
    out.append("boundary p p p\n")
    out.append("\n")

    out.append("#######################\n#   Atom definition   #\n#######################\n\n")
    out.append("\n")
    if continue_run:
        out.append("read_restart star2.restart remap\n")
        out.append("\n")
    else:
        out.append("read_data star.dat\n")
        if colloid:
            out.append("group coll type 3\n")
            out.append("velocity coll set 0.01 0.01 0.01\n")
        out.append("\n")

    out.append("################\n#   Settings   #\n################\n")
    out.append("\n")
    out.append("neighbor 0.8 bin\n")
    if ntot >= 2000:
        out.append("neigh_modify check yes one {}\n".format(ntot + 1))
    else:
        out.append("neigh_modify check yes\n")
    out.append("\n")

    out.append("pair_style hybrid lj/cut {0} lj/sf {1}\n".format(CUT * sigma, CUT * sigma))
    out.append("pair_coeff 1 1 lj/sf 1.0 1.0 {}\n".format(CUT * sigma))
    out.append("pair_coeff 1 2 lj/sf 1.0 {0} {1}\n".format(s_core, CUT * s_core))
    out.append("pair_coeff 2 2 lj/sf 1.0 {0} {1}\n".format(2 * r_core, CUT * 2 * r_core))
    if colloid:
        # Colloid diameter (twice its radius rs / q) and Lorentz mixing.
        sigma_colloid = 2 * (star_radius(arms, ns) / params.q)
        s13 = np.mean([sigma, sigma_colloid])
        s23 = np.mean([2 * r_core, sigma_colloid])
        out.append("pair_coeff 1 3 lj/cut {0} {1} {2}\n".format(params.eps, s13, 2.5 * s13))
        out.append("pair_coeff 2 3 lj/cut {0} {1} {2}\n".format(params.eps, s23, 2.5 * s23))
        out.append("pair_coeff 3 3 lj/sf 1.0 {0} {1}\n".format(sigma_colloid, CUT * sigma_colloid))
    out.append("\n")

    out.append("bond_style fene\n")
    out.append("bond_coeff 1 {0} {1} {2} {3}\n".format(fene_k(sigma), fene_r0(sigma), ENERGY, sigma))
    out.append("bond_coeff 2 {0} {1} {2} {3}\n".format(fene_k(s_core), fene_r0(s_core), ENERGY, s_core))
    out.append("\n")

    out.append("minimize 1e-5 1e-5 1000 1000\n")
    out.append("reset_timestep 0\n")
    out.append("\n")
    out.append("fix 1 all nvt temp {0} {0} 1.0\n".format(TEMPERATURE))
    out.append("\n")

    out.append("#####################\n#   Calculations    #\n#####################\n")
    out.append("\n")
    out.append("group star type 1 2\n")
    out.append("\n")
    # Radius of gyration of the star (types 1 and 2), averaged over 50 samples
    # taken every 1000 steps, written every 500000 steps.
    out.append("compute gyr star gyration\n")
    out.append("fix gyration1 star ave/time 1000 50 500000 c_gyr "
               "file gyration/gyr_all_{}.txt start 500000\n".format(file_number))
    out.append("\n")

    out.append("###########\n# Dumping #\n###########\n")
    out.append("\n")
    out.append("restart 100000 star.restart star2.restart\n")
    out.append("\n")
    out.append("dump config all custom 500000 "
               "configurations/star_{}.lammpstrj id type x y z\n".format(file_number))
    out.append("\n")

    out.append("##########################\n#   Running simulation   #\n##########################\n")
    out.append("\n")
    out.append("timestep 0.01\n")
    out.append("thermo 500000\n")
    out.append("thermo_style custom step temp etotal ke pe epair emol press vol\n")
    out.append("thermo_modify flush yes\n")
    out.append("run 1000000000\n")
    return "".join(out)


def build_input_bbrush(params, file_number, continue_run, seeds):
    """Return the text of ``in.bbrush``.

    Args:
        params: :class:`functions.Params` of a bottlebrush run.
        file_number: index of the output files (as for the star).
        continue_run: True to read ``bbrush2.restart`` instead of ``bbrush.dat``
            (the colloids are then already part of the restart file).
        seeds: ``(seed1, seed2)`` for the two ``create_atoms`` calls.
    """
    sigma = params.sigma
    colloids = bool(params.colloids)
    nb, nbrush, natoms = brush_size(params.nb, params.ns, params.step)
    if colloids:
        natoms += params.ncoll1 + params.ncoll2

    out = []
    out.append("######################\n#   Initialization   #\n######################\n")
    out.append("\n")
    out.append("units lj\n")
    out.append("atom_style bond\n")
    out.append("special_bonds fene\n")
    out.append("boundary p p p\n")
    out.append("\n")

    out.append("#######################\n#   Atom definition   #\n#######################\n\n")
    out.append("\n")
    if continue_run:
        out.append("read_restart bbrush2.restart remap\n")
    else:
        out.append("read_data bbrush.dat\n")
        if colloids:
            out.append("create_atoms 3 random {0} {1} NULL\n".format(params.ncoll1, seeds[0]))
            out.append("create_atoms 4 random {0} {1} NULL\n".format(params.ncoll2, seeds[1]))
    out.append("\n")

    out.append("################\n#   Settings   #\n################\n")
    out.append("\n")
    out.append("neighbor 0.8 bin\n")
    if natoms >= 2000:
        out.append("neigh_modify check yes one {}\n".format(natoms + 1))
    else:
        out.append("neigh_modify check yes\n")
    out.append("\n")

    out.append("pair_style lj/cut {}\n".format(CUT * sigma))
    out.append("pair_coeff 1 1 1.0 {}\n".format(sigma))
    out.append("pair_coeff 1 2 1.0 {}\n".format(sigma))
    out.append("pair_coeff 2 2 1.0 {}\n".format(sigma))
    if colloids:
        d1, d2 = params.dcoll1, params.dcoll2
        for atom_type, diameter in ((3, d1), (4, d2)):
            mixed = np.mean([sigma, diameter])
            for bead in (1, 2):
                out.append("pair_coeff {0} {1} {2} {3} {4}\n".format(
                    bead, atom_type, params.eps, mixed, 2.5 * mixed))
        out.append("pair_coeff 3 3 1.0 {0} {1}\n".format(d1, CUT * d1))
        d12 = np.mean([d1, d2])
        out.append("pair_coeff 3 4 1.0 {0} {1}\n".format(d12, CUT * d12))
        out.append("pair_coeff 4 4 1.0 {0} {1}\n".format(d2, CUT * d2))
    out.append("\n")

    out.append("bond_style fene\n")
    out.append("bond_coeff 1 {0} {1} {2} {3}\n".format(
        float(fene_k(sigma)), fene_r0(sigma), float(ENERGY), sigma))
    out.append("\n")

    out.append("minimize 1e-5 1e-5 1000 1000\n")
    out.append("reset_timestep 0\n")
    out.append("\n")
    out.append("fix 1 all nvt temp {0} {0} 1.0\n".format(TEMPERATURE))
    out.append("\n")

    out.append("#####################\n#   Calculations    #\n#####################\n")
    out.append("\n")
    # Radius of gyration of the backbone and of the whole brush (types 1 and 2).
    out.append("group bbone id 1:{}\n".format(nb))
    out.append("compute gyrb bbone gyration\n")
    out.append("fix gyration bbone ave/time 1000 50 500000 c_gyrb "
               "file gyration/gyr_bbone_{}.txt start 500000\n".format(file_number))
    out.append("\n")
    out.append("group bbrush type 1 2\n")
    out.append("\n")
    out.append("compute gyr bbrush gyration\n")
    out.append("fix gyration1 bbrush ave/time 1000 50 500000 c_gyr "
               "file gyration/gyr_all_{}.txt start 500000\n".format(file_number))
    out.append("\n")
    out.append("group bbone_extremes id 1 {}\n".format(nb))
    out.append("\n")

    out.append("###########\n# Dumping #\n###########\n")
    out.append("\n")
    out.append("restart 250000 bbrush.restart bbrush2.restart\n")
    out.append("\n")
    out.append("dump config all custom 500000 "
               "configurations/bbrush_{}.lammpstrj id type x y z\n".format(file_number))
    out.append("\n")
    out.append("dump endtoend bbone_extremes xyz 500000 "
               "gyration/bbone_extremes_{}.lammpstrj\n".format(file_number))
    out.append("\n")

    out.append("##########################\n#   Running simulation   #\n##########################\n")
    out.append("\n")
    out.append("timestep 0.001\n")
    out.append("thermo 500000\n")
    out.append("thermo_style custom step temp etotal ke pe epair emol press vol\n")
    out.append("thermo_modify flush yes\n")
    out.append("run 1000000000\n")
    return "".join(out)


def build_input(params, file_number, continue_run, seeds=None):
    """Return the input script text for the architecture of ``params``.

    ``seeds`` is only used by the bottlebrush; if None they are derived from
    ``params.seed`` or drawn at random.
    """
    if params.architecture == "star":
        return build_input_star(params, file_number, continue_run)
    if seeds is None:
        if params.seed is not None:
            seeds = (params.seed, params.seed + 1)
        else:
            rng = random.SystemRandom()
            seeds = (rng.randint(1, 2 ** 31 - 2), rng.randint(1, 2 ** 31 - 2))
    return build_input_bbrush(params, file_number, continue_run, seeds)


def main():
    """Read ``par.txt`` and write the input script in the current directory."""
    params = read_par()
    spec = params.spec
    continue_run = params.restart == 1 and os.path.isfile(spec["restart_file"])
    number = check_file_number(params.restart, params.architecture)
    with open(spec["input_file"], "w") as handle:
        handle.write(build_input(params, number, continue_run))


if __name__ == "__main__":
    main()
