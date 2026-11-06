"""Tests for post_processing.py on small synthetic trajectories."""
import os

import numpy as np
import pytest

import functions as fn
import post_processing as pp

LBOX = 100.0
PARAMS = fn.Params(arms=2, ns=2, sigma=1.0, q=2.0, eps=0.2, lbox=LBOX)
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

    hist, tables = pp.analyse(str(tmp_path / "configurations"), PARAMS)

    for h in hist.values():
        assert np.all(np.isfinite(h.y))
    # frame 0 skipped -> near, far, near, far
    assert tables["loading"][1] == [(1, 1), (0, 0), (1, 1), (0, 0)]
    dx = np.diff(hist["coll_distr_anch_2rg"].bins)[0]
    assert hist["coll_distr_anch_2rg"].y.sum() * dx == pytest.approx(1.0)  # averaged over the 2 frames with data
    dx_all = np.diff(hist["coll_distr_anch"].bins)[0]
    assert hist["coll_distr_anch"].y.sum() * dx_all == pytest.approx(1.0)


def test_write_results(tmp_path, monkeypatch):
    """All output files are written with a header and finite numbers."""
    os.makedirs(str(tmp_path / "configurations"))
    text = "".join(straight_frame([60, 50, 50]) for _ in range(3))
    (tmp_path / "configurations" / "star_0.lammpstrj").write_text(text)
    hist, tables = pp.analyse(str(tmp_path / "configurations"), PARAMS)
    out = tmp_path / "analysis"
    pp.write_results(hist, tables, str(out))
    names = sorted(os.listdir(str(out)))
    assert names == ["coll_distr_anch.txt", "coll_distr_anch_2rg.txt", "coll_distr_cm.txt",
                     "loading.txt", "mon_distr_anch.txt", "mon_distr_cm.txt", "rg.txt"]
    assert (out / "rg.txt").read_text().startswith("# x    y\n")


def test_no_frames_raises(tmp_path):
    """An empty configurations folder is an error, not a silent empty result."""
    os.makedirs(str(tmp_path / "configurations"))
    with pytest.raises(RuntimeError):
        pp.analyse(str(tmp_path / "configurations"), PARAMS)


def test_star_without_colloid_analysis(tmp_path):
    """colloids 0: only monomer histograms and Rg are written."""
    params = PARAMS.replace(colloids=0)
    os.makedirs(str(tmp_path / "configurations"))
    frame = straight_frame([60, 50, 50]).replace("6 3 60 50 50\n", "").replace(
        "NUMBER OF ATOMS\n6", "NUMBER OF ATOMS\n5")
    (tmp_path / "configurations" / "star_0.lammpstrj").write_text(frame * 3)
    hist, tables = pp.analyse(str(tmp_path / "configurations"), params)
    assert sorted(hist) == ["mon_distr_anch", "mon_distr_cm", "rg"] and tables == {}


# ---- bottlebrush ---------------------------------------------------------

BB = fn.Params(architecture="bottlebrush", nb=4, ns=2, step=2, lbox=LBOX, eps=0.1)


def bb_frame(shift=0.0, wrap=False, step=0):
    """Straight backbone of 4 beads along y, 2 side beads on one anchor, 1 colloid per species."""
    rows = [(1, 2, 50, 50 + shift, 50), (2, 2, 50, 51 + shift, 50), (3, 2, 50, 52 + shift, 50),
            (4, 2, 50, 53 + shift, 50), (5, 1, 51, 51 + shift, 50), (6, 1, 52, 51 + shift, 50),
            (7, 3, 60, 51.5 + shift, 50), (8, 4, 50, 51.5 + shift, 70)]
    if wrap:   # move the backbone across the periodic boundary in y
        rows = [(i, t, x, (y + 49) % LBOX, z) for i, t, x, y, z in rows]
    lines = ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(rows)),
             "ITEM: BOX BOUNDS pp pp pp"] + ["0.0 {}".format(LBOX)] * 3 + ["ITEM: ATOMS id type x y z"]
    lines += ["{} {} {} {} {}".format(*r) for r in rows]
    return "\n".join(lines) + "\n"


def to_array(text):
    """Parse a one-frame dump text into the frame array."""
    return np.array([[float(x) for x in line.split()] for line in text.split("\n")[9:] if line])


@pytest.mark.parametrize("wrap", [False, True])
def test_bbrush_observables(wrap):
    """End-to-end distance and colloid distances, also with the backbone across the boundary."""
    obs = pp.bbrush_observables(to_array(bb_frame(wrap=wrap)), LBOX)
    assert obs["ree"] == pytest.approx(3.0)
    assert obs["rg_backbone"] == pytest.approx(np.sqrt(np.mean([1.5 ** 2, 0.5 ** 2, 0.5 ** 2, 1.5 ** 2])))
    brush = np.array([[50, 50, 50], [50, 51, 50], [50, 52, 50], [50, 53, 50], [51, 51, 50], [52, 51, 50]], float)
    cm = brush.mean(axis=0)
    assert obs["coll_a_cm"][0] == pytest.approx(np.linalg.norm(np.array([60, 51.5, 50]) - cm))
    assert obs["coll_b_cm"][0] == pytest.approx(np.linalg.norm(np.array([50, 51.5, 70]) - cm))
    assert obs["rg"] == pytest.approx(np.sqrt(np.mean(np.sum((brush - cm) ** 2, axis=1))))


def test_bbrush_analysis_writes_files(tmp_path):
    """Full bottlebrush analysis on a synthetic trajectory (first frame skipped)."""
    os.makedirs(str(tmp_path / "configurations"))
    (tmp_path / "configurations" / "bbrush_0.lammpstrj").write_text(
        "".join(bb_frame(shift=s, step=i) for i, s in enumerate((0, 1, 2, 3))))
    hist, tables = pp.analyse(str(tmp_path / "configurations"), BB)
    assert sorted(hist) == ["coll_a_distr_cm", "coll_b_distr_cm", "mon_distr_cm", "rg"]
    assert len(tables["end_to_end"][1]) == 3
    assert all(np.all(np.isfinite(h.y)) for h in hist.values())
    out = tmp_path / "analysis"
    pp.write_results(hist, tables, str(out))
    assert (out / "end_to_end.txt").read_text().startswith("# frame ree rg_backbone rg\n0 3.0")
