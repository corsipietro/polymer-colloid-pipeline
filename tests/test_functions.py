"""Tests for functions.py."""
import os

import pytest

import functions as fn


def test_read_par_example(tmp_path):
    """The par.txt of the example run is parsed into the expected tuple."""
    path = tmp_path / "par.txt"
    path.write_text("f 10\nns 50\nsigma 1\nlbox 161.5\nq 1.8\neps 0.2\nrestart 1")
    assert fn.read_par(str(path)) == fn.Params(
        arms=10, ns=50, sigma=1.0, q=1.8, eps=0.2, box=[161.5] * 3, restart=1, rcore=1.75)


def test_read_par_missing_key(tmp_path):
    """A missing key is reported by name."""
    path = tmp_path / "par.txt"
    path.write_text("f 10\n")
    with pytest.raises(KeyError, match="lbox"):
        fn.read_par(str(path))


def test_write_par_roundtrip(tmp_path):
    """write_par followed by read_par returns the same values."""
    path = str(tmp_path / "par.txt")
    fn.write_par(path, 50, 100, 1, 123.456789012345, 1.4, 0.05, 0, rcore=2.25)
    p = fn.read_par(path)
    assert (p.arms, p.ns, p.sigma, p.q, p.eps, p.restart) == (50, 100, 1.0, 1.4, 0.05, 0)
    assert p.rcore == 2.25
    assert p.box == [123.456789012345] * 3


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
    """Only q_* leaves are returned, optionally filtered by exact ns."""
    for ns in (100, 1000):
        os.makedirs(str(tmp_path / "f_50" / "ns_{}".format(ns) / "eps_0.2" / "q_1.8"))
    os.makedirs(str(tmp_path / "f_50" / "ns_100" / "eps_0.2"), exist_ok=True)
    assert len(fn.find_run_dirs(str(tmp_path))) == 2
    only = fn.find_run_dirs(str(tmp_path), ns=100)
    assert len(only) == 1 and "ns_100" + os.sep in only[0]


def test_distance_and_radius():
    """3-4-5 triangle and the star radius formula."""
    assert fn.distance([0, 0, 0], [3, 4, 0]) == 5.0
    assert fn.star_radius(1, 1) == pytest.approx(0.6)
