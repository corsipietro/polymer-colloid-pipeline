"""Tests for functions.py."""
import os

import pytest

import functions as fn


def test_read_par_legacy_star_file(tmp_path):
    """A par.txt without architecture/mode/colloids is read as a star run with a colloid."""
    path = tmp_path / "par.txt"
    path.write_text("f 10\nns 50\nsigma 1\nlbox 161.5\nq 1.8\neps 0.2\nrestart 1")
    p = fn.read_par(str(path))
    assert (p.architecture, p.mode, p.present) == ("star", "cluster", (1, 0))
    assert (p.arms, p.ns, p.sigma, p.q, p.eps, p.restart, p.rcore) == (10, 50, 1.0, 1.8, 0.2, 1, 1.75)
    assert p.box == [161.5] * 3 and p.spec["input_file"] == "in.star"


def test_read_par_bottlebrush_defaults(tmp_path):
    """Bottlebrush without colloid keys: two species of 40, explicit diameters 14 and 10, eps 0.1."""
    p = fn.read_par(os.path.join(os.path.dirname(__file__), "data", "par_bbrush.txt"))
    assert (p.architecture, p.mode, p.nb, p.ns, p.step) == ("bottlebrush", "local", 99, 50, 3)
    assert (p.eps, p.n_species, p.counts, p.dcoll1, p.dcoll2) == (0.1, 2, (40, 40), 14.0, 10.0)
    assert p.spec["restart_file"] == "bbrush2.restart"


def test_colloid_species_counts(tmp_path):
    """colloids 0/1/2 select the species that are present."""
    star = fn.Params(arms=5, ns=5, lbox=50.0, q=2.0, q2=3.0, eps=0.1)
    assert star.species() == [(1, 3, 1)]                      # star default: one colloid
    assert star.replace(colloids=0).species() == [] and star.replace(colloids=0).n_colloids == 0
    two = star.replace(colloids=2, ncoll1=2, ncoll2=4)
    assert two.species() == [(1, 3, 2), (2, 4, 4)] and two.n_colloids == 6
    brush = fn.Params(architecture="bottlebrush", nb=9, ns=3, step=3, lbox=50.0)
    assert brush.species() == [(1, 3, 40), (2, 4, 40)]         # bottlebrush default: two species
    assert brush.replace(colloids=1).species() == [(1, 3, 40)]


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


def test_star_two_species_needs_q2(tmp_path):
    """colloids 2 on a star requires q2."""
    path = tmp_path / "par.txt"
    path.write_text("colloids 2\nf 10\nns 50\nsigma 1\nlbox 100\nq 1.8\neps 0.2\nrestart 0\n")
    with pytest.raises(KeyError, match="q2"):
        fn.read_par(str(path))
    path.write_text("colloids 3\nf 10\nns 50\nsigma 1\nlbox 100\nrestart 0\n")
    with pytest.raises(ValueError, match="colloids"):
        fn.read_par(str(path))


def test_star_without_colloid_does_not_need_q_and_eps(tmp_path):
    """colloids 0 drops the colloid keys."""
    path = tmp_path / "par.txt"
    path.write_text("colloids 0\nf 10\nns 50\nsigma 1\nlbox 100\nrestart 0\n")
    p = fn.read_par(str(path))
    assert p.present == (0, 0) and p.q is None


@pytest.mark.parametrize("params", [
    fn.Params(architecture="star", mode="local", arms=50, ns=100, q=1.4, eps=0.05, colloids=1,
              lbox=123.456789012345, rcore=2.25, seed=7),
    fn.Params(architecture="star", arms=20, ns=30, q=1.4, q2=2.2, eps=0.1, eps2=0.3, colloids=2,
              ncoll1=2, ncoll2=3, lbox=90.0),
    fn.Params(architecture="star", colloids=0, arms=10, ns=20, lbox=80.0),
    fn.Params(architecture="star", colloid1=0, colloid2=1, arms=10, ns=20, lbox=80.0, q2=2.0, eps=0.1,
              ncoll2=3),
    fn.Params(architecture="bottlebrush", colloid1=0, colloid2=1, nb=30, ns=10, step=2, lbox=60.0,
              eps=0.1, q2=2.0),
    fn.Params(architecture="bottlebrush", mode="local", nb=99, ns=50, step=3, lbox=120.0,
              eps=0.2, colloids=2, ncoll1=5, dcoll1=14.0, ncoll2=6, dcoll2=10.0, restart=1),
    fn.Params(architecture="bottlebrush", nb=99, ns=50, step=3, lbox=120.0, eps=0.1,
              colloids=2, ncoll1=1, ncoll2=2, q=2.5, q2=1.5),
    fn.Params(architecture="bottlebrush", colloids=1, nb=30, ns=10, step=2, lbox=60.0, eps=0.1, q=2.0),
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
        fn.write_par(str(run / "par.txt"), fn.Params(arms=50, ns=ns, q=1.8, eps=0.2, colloids=1, lbox=100.0))
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


def test_each_colloid_species_is_independent(tmp_path):
    """colloid1 / colloid2 choose species 1 only, species 2 only, both or none (star and bottlebrush)."""
    star = "architecture star\nf 10\nns 50\nsigma 1\nlbox 100\nrestart 0\neps 0.2\n"
    brush = "architecture bottlebrush\nnb 30\nns 8\nstep 3\nsigma 1\nlbox 60\nrestart 0\n"
    path = tmp_path / "par.txt"
    for base, extra in ((star, "q 1.8\nq2 2.6\n"), (brush, "")):
        for c1, c2 in ((0, 0), (1, 0), (0, 1), (1, 1)):
            path.write_text(base + extra + "colloid1 {}\ncolloid2 {}\n".format(c1, c2))
            p = fn.read_par(str(path))
            assert p.present == (c1, c2)
            assert [i for i, _, _ in p.species()] == [i for i, on in ((1, c1), (2, c2)) if on]
            assert p.n_species == (2 if c2 else 1 if c1 else 0)      # declared types stay contiguous
    # a star needs q only for the species that is present
    path.write_text(star + "q2 2.6\ncolloid1 0\ncolloid2 1\n")
    assert fn.read_par(str(path)).q is None
    path.write_text(star + "q 1.8\ncolloid1 1\ncolloid2 1\n")
    with pytest.raises(KeyError, match="q2"):
        fn.read_par(str(path))
    path.write_text(star + "q 1.8\ncolloid1 3\n")
    with pytest.raises(ValueError, match="colloid1"):
        fn.read_par(str(path))


def test_legacy_colloids_key_still_understood(tmp_path):
    """colloids 0/1/2 (number of species) maps to colloid1 / colloid2."""
    path = tmp_path / "par.txt"
    for n, flags in ((0, (0, 0)), (1, (1, 0)), (2, (1, 1))):
        path.write_text("architecture bottlebrush\ncolloids {}\nnb 30\nns 8\nstep 3\nsigma 1\n"
                        "lbox 60\nrestart 0\n".format(n))
        assert fn.read_par(str(path)).present == flags
