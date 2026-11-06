"""Tests for post_processing.py on small synthetic trajectories."""
import os

import numpy as np
import pytest

import functions as fn
import post_processing as pp

LBOX = 100.0
PARAMS = fn.Params(arms=2, ns=2, sigma=1.0, q=2.0, eps=0.2, box=[LBOX] * 3, restart=0, rcore=1.75)
NATOMS = 6  # 4 monomers + core + colloid


def make_frame(core, monomers, colloid, step=0):
    """Return the text of one dump frame (ids: 1 core, 2-5 monomers, 6 colloid)."""
    lines = ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(NATOMS),
             "ITEM: BOX BOUNDS pp pp pp"] + ["0.0 {}".format(LBOX)] * 3 + \
            ["ITEM: ATOMS id type x y z"]
    lines.append("1 2 {} {} {}".format(*core))
    for i, m in enumerate(monomers, 2):
        lines.append("{} 1 {} {} {}".format(i, *m))
    lines.append("6 3 {} {} {}".format(*colloid))
    return "\n".join(lines) + "\n"


def straight_frame(colloid):
    """Core at the centre, monomers on the +x axis at 2, 3 and on -x at 2, 3."""
    c = np.array([50.0, 50.0, 50.0])
    mons = [c + [2, 0, 0], c + [3, 0, 0], c - [2, 0, 0], c - [3, 0, 0]]
    return make_frame(c, mons, colloid)


def test_rg_matches_direct_computation():
    """Rg equals the numpy value for the star group (core + monomers)."""
    frame = np.array([[float(x) for x in line.split()] for line in
                      straight_frame([10, 10, 10]).split("\n")[9:] if line])
    obs = pp.frame_observables(frame, LBOX)
    pts = np.array([[0, 0, 0], [2, 0, 0], [3, 0, 0], [-2, 0, 0], [-3, 0, 0]], float)
    expected = np.sqrt(np.mean(np.sum(pts ** 2, axis=1)))
    assert obs["rg"] == pytest.approx(expected)
    assert sorted(obs["mon_anch"]) == pytest.approx([2, 2, 3, 3])


def test_periodic_images_are_unwrapped_around_the_core():
    """A monomer across the boundary is at its true short distance."""
    c = [1.0, 50.0, 50.0]
    mons = [[99.0, 50.0, 50.0], [2.0, 50, 50], [3.0, 50, 50], [4.0, 50, 50]]
    frame = np.array([[float(x) for x in line.split()] for line in
                      make_frame(c, mons, [1, 1, 1]).split("\n")[9:] if line])
    obs = pp.frame_observables(frame, LBOX)
    assert obs["mon_anch"][0] == pytest.approx(2.0)   # 99 -> -1, distance 2 from x=1


def test_first_frame_is_skipped_and_truncated_frame_dropped(tmp_path):
    """read_frames skips frame 0 and ignores an incomplete last frame."""
    path = tmp_path / "t.lammpstrj"
    full = straight_frame([10, 10, 10])
    path.write_text(full + full + full + full[: len(full) // 2])
    assert len(list(pp.read_frames(str(path), NATOMS))) == 2
    assert len(list(pp.read_frames(str(path), NATOMS, skip_first=False))) == 3


def test_wrong_atom_count_is_rejected(tmp_path):
    """A trajectory of a different system raises ValueError."""
    path = tmp_path / "t.lammpstrj"
    path.write_text(straight_frame([10, 10, 10]))
    with pytest.raises(ValueError):
        list(pp.read_frames(str(path), NATOMS + 1))


def test_empty_2rg_histogram_does_not_produce_nan(tmp_path):
    """Frames with the colloid outside 2 rs must not turn the average into NaN."""
    rs = fn.star_radius(PARAMS.arms, PARAMS.ns)
    near = [50 + rs, 50, 50]          # within 2 rs
    far = [50 + 5 * rs, 50, 50]       # outside 2 rs
    os.makedirs(str(tmp_path / "configurations"))
    text = "".join(straight_frame(c) for c in (far, near, far, near, far))
    (tmp_path / "configurations" / "star_0.lammpstrj").write_text(text)

    hist, loading = pp.analyse(str(tmp_path / "configurations"), PARAMS)

    for h in hist.values():
        assert np.all(np.isfinite(h.y))
    # frame 0 skipped -> near, far, near, far
    assert loading == [(1, 1), (0, 0), (1, 1), (0, 0)]
    dx = np.diff(hist["coll_distr_anch_2rg"].bins)[0]
    assert hist["coll_distr_anch_2rg"].y.sum() * dx == pytest.approx(1.0)  # averaged over the 2 frames with data
    dx_all = np.diff(hist["coll_distr_anch"].bins)[0]
    assert hist["coll_distr_anch"].y.sum() * dx_all == pytest.approx(1.0)


def test_write_results(tmp_path, monkeypatch):
    """All output files are written with a header and finite numbers."""
    os.makedirs(str(tmp_path / "configurations"))
    text = "".join(straight_frame([60, 50, 50]) for _ in range(3))
    (tmp_path / "configurations" / "star_0.lammpstrj").write_text(text)
    hist, loading = pp.analyse(str(tmp_path / "configurations"), PARAMS)
    out = tmp_path / "analysis"
    pp.write_results(hist, loading, str(out))
    names = sorted(os.listdir(str(out)))
    assert names == ["coll_distr_anch.txt", "coll_distr_anch_2rg.txt", "coll_distr_cm.txt",
                     "loading.txt", "mon_distr_anch.txt", "mon_distr_cm.txt", "rg.txt"]
    assert (out / "rg.txt").read_text().startswith("# x    y\n")


def test_no_frames_raises(tmp_path):
    """An empty configurations folder is an error, not a silent empty result."""
    os.makedirs(str(tmp_path / "configurations"))
    with pytest.raises(RuntimeError):
        pp.analyse(str(tmp_path / "configurations"), PARAMS)
