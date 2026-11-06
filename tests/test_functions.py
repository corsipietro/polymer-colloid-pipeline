"""Tests for functions.py."""
import os

import pytest

import functions as fn


def test_read_par_legacy_star_file(tmp_path):
    """A par.txt without architecture/mode/colloids is read as a star run with a colloid."""
    path = tmp_path / "par.txt"
    path.write_text("f 10\nns 50\nsigma 1\nlbox 161.5\nq 1.8\neps 0.2\nrestart 1")
    p = fn.read_par(str(path))
    assert (p.architecture, p.mode, p.colloids) == ("star", "cluster", 1)
    assert (p.arms, p.ns, p.sigma, p.q, p.eps, p.restart, p.rcore) == (10, 50, 1.0, 1.8, 0.2, 1, 1.75)
    assert p.box == [161.5] * 3 and p.spec["input_file"] == "in.star"


def test_read_par_bottlebrush_defaults(tmp_path):
    """Bottlebrush: colloid parameters default to the original in.bbrush values."""
    p = fn.read_par(os.path.join(os.path.dirname(__file__), "data", "par_bbrush.txt"))
    assert (p.architecture, p.mode, p.nb, p.ns, p.step) == ("bottlebrush", "local", 99, 50, 3)
    assert (p.eps, p.ncoll1, p.dcoll1, p.ncoll2, p.dcoll2) == (0.1, 40, 14.0, 40, 10.0)
    assert p.spec["restart_file"] == "bbrush2.restart"


def test_read_par_errors(tmp_path):
    """Missing keys, unknown architecture and unknown mode are reported."""
    path = tmp_path / "par.txt"
    path.write_text("f 10\n")
    with pytest.raises(KeyError, match="lbox"):
        fn.read_par(str(path))
    path.write_text("architecture polymer\n")
    with pytest.raises(ValueError, match="architecture"):
        fn.read_par(str(path))
    path.write_text("mode gpu\n")
    with pytest.raises(ValueError, match="mode"):
        fn.read_par(str(path))


def test_star_without_colloid_does_not_need_q_and_eps(tmp_path):
    """colloids 0 drops the colloid keys."""
    path = tmp_path / "par.txt"
    path.write_text("colloids 0\nf 10\nns 50\nsigma 1\nlbox 100\nrestart 0\n")
    p = fn.read_par(str(path))
    assert p.colloids == 0 and p.q is None


@pytest.mark.parametrize("params", [
    fn.Params(architecture="star", mode="local", arms=50, ns=100, q=1.4, eps=0.05,
              lbox=123.456789012345, rcore=2.25, seed=7),
    fn.Params(architecture="star", colloids=0, arms=10, ns=20, lbox=80.0),
    fn.Params(architecture="bottlebrush", mode="local", nb=99, ns=50, step=3, lbox=120.0,
              eps=0.2, ncoll1=5, dcoll1=14.0, ncoll2=6, dcoll2=10.0, restart=1),
    fn.Params(architecture="bottlebrush", nb=99, ns=50, step=3, lbox=120.0, eps=0.1,
              ncoll1=1, ncoll2=0, q=2.5, q2=1.5),
    fn.Params(architecture="bottlebrush", colloids=0, nb=30, ns=10, step=2, lbox=60.0, eps=0.1),
])
def test_write_par_roundtrip(tmp_path, params):
    """write_par followed by read_par returns the same parameters."""
    path = str(tmp_path / "par.txt")
    fn.write_par(path, params)
    assert fn.read_par(path) == params


def test_check_file_number(tmp_path, monkeypatch):
    """The next index follows the highest existing one, also beyond 9."""
    monkeypatch.chdir(tmp_path)
    os.mkdir("gyration")
    os.mkdir("configurations")
    assert fn.check_file_number(1) == 0          # no restart file yet
    (tmp_path / "star2.restart").write_text("x")
    assert fn.check_file_number(0) == 0          # not a restart
    assert fn.check_file_number(1) == 0          # restart file but no output
    for n in (0, 1, 9, 10):
        (tmp_path / "gyration" / "gyr_all_{}.txt".format(n)).write_text("x")
    assert fn.check_file_number(1) == 11         # old code returned 2 here


def test_find_run_dirs(tmp_path):
    """Runs are the directories with a par.txt, optionally filtered by ns."""
    for ns in (100, 1000):
        run = tmp_path / "f_50" / "ns_{}".format(ns) / "eps_0.2" / "q_1.8"
        os.makedirs(str(run))
        fn.write_par(str(run / "par.txt"), fn.Params(arms=50, ns=ns, q=1.8, eps=0.2, lbox=100.0))
    assert len(fn.find_run_dirs(str(tmp_path))) == 2
    only = fn.find_run_dirs(str(tmp_path), ns=100)
    assert len(only) == 1 and "ns_100" + os.sep in only[0]


def test_check_file_number_bottlebrush(tmp_path, monkeypatch):
    """The bottlebrush counts bbrush_<n> files and its own restart file."""
    monkeypatch.chdir(tmp_path)
    os.mkdir("configurations")
    (tmp_path / "bbrush2.restart").write_text("x")
    (tmp_path / "configurations" / "bbrush_3.lammpstrj").write_text("x")
    (tmp_path / "configurations" / "star_9.lammpstrj").write_text("x")
    assert fn.check_file_number(1, "bottlebrush") == 4


def test_distance_and_radius():
    """3-4-5 triangle and the star radius formula."""
    assert fn.distance([0, 0, 0], [3, 4, 0]) == 5.0
    assert fn.star_radius(1, 1) == pytest.approx(0.6)
