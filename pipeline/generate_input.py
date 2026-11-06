#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write the LAMMPS input script ``in.star`` for one run.

Interactions (LJ units, reduced temperature 1.0):
  * monomer-monomer, monomer-core, core-core: purely repulsive (shifted LJ,
    cut at the minimum 2^(1/6) * sigma_ij);
  * colloid-monomer and colloid-core: attractive LJ of depth ``eps``, cut at
    2.5 sigma_ij;
  * colloid-colloid: repulsive, with diameter ``2 * rs / q`` (``rs`` = star
    radius, ``q`` = star/colloid size ratio);
  * FENE bonds (k = 30 / s^2, R0 = 1.5 s).

A fresh run reads ``star.dat``; with ``restart 1`` and an existing
``star2.restart`` the run is continued, writing to the next output file index.

Usage (from a run directory containing ``par.txt``)::

    python3 generate_input.py
"""

import os

import numpy as np

from functions import check_file_number, read_par, star_radius

# LJ well depth for the bond/pair energy scale and the thermostat temperature.
ENERGY = 1
TEMPERATURE = "1.0"


def build_input(params, file_number, continue_run):
    """Return the text of ``in.star``.

    Args:
        params: :class:`functions.Params` of the run.
        file_number: index used in the names of the output files
            (``gyration/gyr_all_<n>.txt``, ``configurations/star_<n>.lammpstrj``).
        continue_run: True to read ``star2.restart`` instead of ``star.dat``.
    """
    arms, ns, sigma, q, eps = params.arms, params.ns, params.sigma, params.q, params.eps
    r_core = params.rcore
    ntot = 1 + ns * arms

    # Colloid diameter and effective core-monomer distance.
    sigma_colloid = 2 * (star_radius(arms, ns) / q)
    s_core = r_core + sigma / 2
    cut = 2 ** (1 / 6)  # minimum of the LJ potential, in units of sigma_ij

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

    out.append("pair_style hybrid lj/cut {0} lj/sf {1}\n".format(cut * sigma, cut * sigma))
    out.append("pair_coeff 1 1 lj/sf 1.0 1.0 {}\n".format(cut * sigma))
    out.append("pair_coeff 1 2 lj/sf 1.0 {0} {1}\n".format(s_core, cut * s_core))
    out.append("pair_coeff 2 2 lj/sf 1.0 {0} {1}\n".format(2 * r_core, cut * 2 * r_core))

    # Lorentz-Berthelot mixing for the colloid cross terms.
    s13 = np.mean([sigma, sigma_colloid])
    s23 = np.mean([2 * r_core, sigma_colloid])
    out.append("pair_coeff 1 3 lj/cut {0} {1} {2}\n".format(eps, s13, 2.5 * s13))
    out.append("pair_coeff 2 3 lj/cut {0} {1} {2}\n".format(eps, s23, 2.5 * s23))
    out.append("pair_coeff 3 3 lj/sf 1.0 {0} {1}\n".format(sigma_colloid, cut * sigma_colloid))
    out.append("\n")

    def fene_k(x):
        """FENE spring constant for a bond of length scale ``x``."""
        return 30 * ENERGY / (x ** 2)

    def fene_r0(x):
        """FENE maximum extension for a bond of length scale ``x``."""
        return 1.5 * x

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


def main():
    """Read ``par.txt`` and write ``in.star`` in the current directory."""
    params = read_par()
    continue_run = params.restart == 1 and os.path.isfile("star2.restart")
    text = build_input(params, check_file_number(params.restart), continue_run)
    with open("in.star", "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
