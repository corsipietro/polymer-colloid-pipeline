#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start a run or an analysis either locally or on the cluster.

Used by ``cluster_submit.py``, ``cluster_restart.py`` and
``post_cluster_submit.py``; it is not copied into the run directories.

* ``cluster``: runs the ``.csh`` script, which submits a PBS job.
* ``local``: simulations are started in the background (they last "forever":
  ``run 1000000000``), with their output in ``run.log``; analyses run in the
  foreground. Stop a local simulation with ``kill`` or Ctrl-C of its process
  group. Running several simulations at once on one machine is slow, so start
  them one at a time.
"""

import os
import subprocess
import sys

from functions import MODES

# Files every run directory needs.
RUN_FILES = ["run_simulation.py", "generate_system.py", "generate_star.py",
             "generate_bottle_brush.py", "generate_input.py", "functions.py",
             "start_py.csh"]
POST_FILES = ["post_processing.py", "functions.py", "generate_bottle_brush.py", "post_py.csh"]


def start_simulation(run_dir, mode):
    """Start the simulation of ``run_dir``; return immediately.

    Raises:
        ValueError: for an unknown mode.
    """
    if mode == "cluster":
        subprocess.call(["./start_py.csh"], cwd=run_dir)
    elif mode == "local":
        log = open(os.path.join(run_dir, "run.log"), "a")
        subprocess.Popen([sys.executable, "run_simulation.py"], cwd=run_dir,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    else:
        raise ValueError("unknown mode '{}' (use one of: {})".format(mode, ", ".join(MODES)))


def start_analysis(run_dir, mode):
    """Run the trajectory analysis of ``run_dir`` (blocking if ``mode`` is local)."""
    if mode == "cluster":
        subprocess.call(["./post_py.csh"], cwd=run_dir)
    elif mode == "local":
        subprocess.call([sys.executable, "post_processing.py"], cwd=run_dir)
    else:
        raise ValueError("unknown mode '{}' (use one of: {})".format(mode, ", ".join(MODES)))
