"""End-to-end checks of the command-line scripts in a scratch run directory."""
import os
import shutil
import subprocess
import sys

import pytest

PIPE = os.path.join(os.path.dirname(__file__), "..", "pipeline")
DATA = os.path.join(os.path.dirname(__file__), "data")


def run(script, cwd, *args):
    """Run a pipeline script in ``cwd`` and return the completed process."""
    return subprocess.run([sys.executable, os.path.join(PIPE, script)] + list(args),
                          cwd=str(cwd), check=True, capture_output=True, text=True)


def test_star_run_directory(tmp_path):
    """par.txt -> star.dat + in.star through the single entry point."""
    shutil.copy(os.path.join(DATA, "par_f10_ns50.txt"), str(tmp_path / "par.txt"))
    run("generate_system.py", tmp_path, "--seed", "3")
    run("generate_input.py", tmp_path)
    assert (tmp_path / "star.dat").read_text().startswith("LAMMPS data file")
    assert "read_data star.dat" in (tmp_path / "in.star").read_text()
    for d in ("gyration", "configurations", "analysis"):
        assert (tmp_path / d).is_dir()


def test_bottlebrush_run_directory(tmp_path):
    """The same entry points build bbrush.dat and in.bbrush."""
    shutil.copy(os.path.join(DATA, "par_bbrush.txt"), str(tmp_path / "par.txt"))
    run("generate_system.py", tmp_path, "--seed", "3")
    run("generate_input.py", tmp_path)
    assert "1749 atoms" in (tmp_path / "bbrush.dat").read_text()
    assert "read_data bbrush.dat" in (tmp_path / "in.bbrush").read_text()


def test_run_simulation_calls_lammps_command(tmp_path):
    """run_simulation.py generates the files then calls $LAMMPS_CMD -in <input>."""
    shutil.copy(os.path.join(DATA, "par_f10_ns50.txt"), str(tmp_path / "par.txt"))
    fake = tmp_path / "fake_lmp.sh"
    fake.write_text("#!/bin/sh\necho \"$@\" > called.txt\n")
    fake.chmod(0o755)
    env = dict(os.environ, LAMMPS_CMD=str(fake))
    subprocess.run([sys.executable, os.path.join(PIPE, "run_simulation.py")], cwd=str(tmp_path),
                   check=True, env=env, capture_output=True, text=True)
    assert (tmp_path / "called.txt").read_text().strip() == "-in in.star"
    assert (tmp_path / "star.dat").exists() and (tmp_path / "in.star").exists()


def test_submit_dry_run_and_local_guard(tmp_path):
    """cluster_submit --dry-run builds the tree; local mode refuses a multi-run scan."""
    done = subprocess.run([sys.executable, os.path.join(PIPE, "cluster_submit.py"), "--dry-run"],
                          cwd=str(tmp_path), capture_output=True, text=True)
    assert done.returncode == 0
    assert len(list(tmp_path.glob("*/f_50/ns_100/eps_*/q_*/par.txt"))) == 35
    refused = subprocess.run([sys.executable, os.path.join(PIPE, "cluster_submit.py"),
                              "--mode", "local"], cwd=str(tmp_path), capture_output=True, text=True)
    assert refused.returncode != 0 and "not recommended" in refused.stderr


@pytest.mark.parametrize("architecture, c1, c2, expected", [
    ("star", 0, 0, 1), ("star", 1, 0, 7 * 5), ("star", 0, 1, 5), ("star", 1, 1, 7 * 5),
    ("bottlebrush", 0, 0, 1), ("bottlebrush", 1, 0, 1), ("bottlebrush", 0, 1, 1), ("bottlebrush", 1, 1, 1)])
def test_scan_runs_for_every_architecture_and_species(monkeypatch, architecture, c1, c2, expected):
    """The scan yields valid par.txt content for each choice of colloid species."""
    sys.path.insert(0, PIPE)
    import cluster_submit as cs
    monkeypatch.setattr(cs, "ARCHITECTURE", architecture)
    monkeypatch.setattr(cs, "COLLOID1", c1)
    monkeypatch.setattr(cs, "COLLOID2", c2)
    runs = list(cs.scan_runs("local"))
    assert len(runs) == expected
    for relative, params in runs:
        assert params.present == (c1, c2) and params.mode == "local"
        assert ("q2_" in relative) == bool(c2 and architecture == "star")
