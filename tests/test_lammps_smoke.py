"""Short real LAMMPS runs of every variant (polymer alone, 1 and 2 colloid species), followed by the analysis.

Skipped if LAMMPS is not available: set ``LAMMPS_CMD`` (default ``lmp``) to run them.
The open-ended ``run 1000000000`` is shortened to 40000 steps by a small wrapper.
"""
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

PIPE = os.path.join(os.path.dirname(__file__), "..", "pipeline")
LMP = shlex.split(os.environ.get("LAMMPS_CMD", "lmp"))
pytestmark = pytest.mark.skipif(shutil.which(LMP[0]) is None and not os.path.isfile(LMP[0]),
                                reason="LAMMPS not found (set LAMMPS_CMD)")

WRAPPER = '''
import re, subprocess, sys
lmp, infile = sys.argv[1:-2], sys.argv[-1]
text = open(infile).read()
text = text.replace("run 1000000000", "run 40000")
text = text.replace("ave/time 1000 50 500000", "ave/time 100 10 5000").replace("start 500000", "start 5000")
text = re.sub(r"(dump \\w+ \\w+ \\w+) 500000", r"\\1 5000", text).replace("thermo 500000", "thermo 5000")
text = text.replace("dump endtoend bbone_extremes xyz 500000", "dump endtoend bbone_extremes xyz 5000")
open(infile, "w").write(text)
sys.exit(subprocess.call(lmp + ["-in", infile]))
'''

BASE = "sigma 1\nrestart 0\nseed 3\n"
RUNS = {
    "star_alone": "architecture star\ncolloid1 0\ncolloid2 0\nf 8\nns 15\nlbox 50\n",
    "star_one_colloid": "architecture star\ncolloid1 1\ncolloid2 0\nf 8\nns 15\nlbox 50\nq 1.8\neps 0.2\n",
    "star_two_species": "architecture star\ncolloid1 1\ncolloid2 1\nf 8\nns 15\nlbox 70\nq 1.8\n"
                        "q2 2.6\neps 0.2\nncoll1 1\nncoll2 2\n",
    "star_species2_only": "architecture star\ncolloid1 0\ncolloid2 1\nf 8\nns 15\nlbox 70\nq2 2.6\n"
                          "eps 0.2\nncoll2 2\n",
    "bbrush_alone": "architecture bottlebrush\ncolloid1 0\ncolloid2 0\nnb 30\nns 8\nstep 3\nlbox 60\n",
    "bbrush_one_species_q": "architecture bottlebrush\ncolloid1 1\ncolloid2 0\nnb 30\nns 8\nstep 3\nlbox 60\n"
                            "eps 0.1\nncoll1 2\nq 2.0\n",
    "bbrush_species2_only": "architecture bottlebrush\ncolloid1 0\ncolloid2 1\nnb 30\nns 8\nstep 3\n"
                            "lbox 60\neps 0.1\nncoll2 3\ndcoll2 6\n",
    "bbrush_two_species": "architecture bottlebrush\ncolloid1 1\ncolloid2 1\nnb 30\nns 8\nstep 3\nlbox 60\n"
                          "eps 0.1\nncoll1 3\ndcoll1 8\nncoll2 3\ndcoll2 6\n",
}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_short_run_and_analysis(tmp_path, name):
    """The whole chain runs: generate, simulate (T close to 1, no lost atoms), analyse."""
    for script in os.listdir(PIPE):
        if script.endswith(".py"):
            shutil.copy(os.path.join(PIPE, script), str(tmp_path))
    (tmp_path / "wrapper.py").write_text(WRAPPER)
    (tmp_path / "par.txt").write_text(RUNS[name] + BASE)
    env = dict(os.environ, LAMMPS_CMD=" ".join(
        [shlex.quote(sys.executable), "wrapper.py"] + [shlex.quote(a) for a in LMP]))
    done = subprocess.run([sys.executable, "run_simulation.py"], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Lost atoms" not in done.stdout

    # last thermo line: step temp ... ; the thermostat brings T to ~1
    rows = [l.split() for l in done.stdout.splitlines() if l.split()[:1] == ["40000"]]
    assert abs(float(rows[-1][1]) - 1.0) < 0.5

    subprocess.run([sys.executable, "post_processing.py"], cwd=str(tmp_path), check=True)
    files = os.listdir(str(tmp_path / "analysis"))
    assert files and all((tmp_path / "analysis" / f).stat().st_size > 0 for f in files)
    for f in files:
        data = np.genfromtxt(str(tmp_path / "analysis" / f), comments="#")
        assert np.all(np.isfinite(data)), f
