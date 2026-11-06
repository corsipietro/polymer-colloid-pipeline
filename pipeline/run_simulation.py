#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the system and the input script of a run, then run LAMMPS.

This is the single execution path used both locally and by the cluster job
(``start_py.csh``). The LAMMPS command comes from the environment variable
``LAMMPS_CMD`` (default ``lmp``); it may include a launcher, e.g.
``LAMMPS_CMD="mpirun -np 4 lmp"``. The command is called as
``<LAMMPS_CMD> -in <in.star|in.bbrush>``.

Usage (from a run directory containing ``par.txt``)::

    python3 run_simulation.py
"""

import os
import shlex
import subprocess
import sys

from functions import read_par


def main():
    """Run the three steps; return the exit status of the first one that fails."""
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ("generate_system.py", "generate_input.py"):
        status = subprocess.call([sys.executable, os.path.join(here, script)])
        if status != 0:
            return status
    params = read_par()
    command = shlex.split(os.environ.get("LAMMPS_CMD", "lmp"))
    return subprocess.call(command + ["-in", params.spec["input_file"]])


if __name__ == "__main__":
    sys.exit(main())
