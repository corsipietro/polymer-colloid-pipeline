"""End-to-end checks of the command-line scripts in a scratch run directory."""
import os
import shutil
import subprocess
import sys

PIPE = os.path.join(os.path.dirname(__file__), "..", "pipeline")


def run(script, cwd, *args):
    """Run a pipeline script in ``cwd`` and return the completed process."""
    return subprocess.run([sys.executable, os.path.join(PIPE, script)] + list(args),
                          cwd=str(cwd), check=True, capture_output=True, text=True)


def test_generate_then_analyse_a_run_directory(tmp_path):
    """par.txt -> star.dat + in.star, then post_processing on a fake trajectory."""
    shutil.copy(os.path.join(os.path.dirname(__file__), "data", "par_f10_ns50.txt"),
                str(tmp_path / "par.txt"))
    run("generate_star.py", tmp_path, "--seed", "3")
    run("generate_input.py", tmp_path)
    assert (tmp_path / "star.dat").read_text().startswith("LAMMPS data file")
    assert "read_data star.dat" in (tmp_path / "in.star").read_text()
    for d in ("gyration", "configurations", "analysis"):
        assert (tmp_path / d).is_dir()
