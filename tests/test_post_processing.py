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


def test_star_two_colloid_species(tmp_path):
    """Two species: files with the _b suffix, loading counts per species."""
    params = fn.Params(arms=2, ns=2, lbox=LBOX, q=2.0, q2=3.0, eps=0.1, colloids=2, ncoll1=1, ncoll2=2)
    rs = fn.star_radius(2, 2)
    base = straight_frame([50 + rs, 50, 50])            # type 3 colloid within rs of the core
    extra = "7 4 {} 50 50\n8 4 50 {} 50\n".format(50 + 5 * rs, 50 + 0.5 * rs)   # one far, one near
    frame = base.replace("NUMBER OF ATOMS\n6", "NUMBER OF ATOMS\n8") + extra
    os.makedirs(str(tmp_path / "configurations"))
    (tmp_path / "configurations" / "star_0.lammpstrj").write_text(frame * 3)
    hist, tables = pp.analyse(str(tmp_path / "configurations"), params)
    assert "coll_distr_anch_b" in hist and "coll_distr_cm_b" in hist and "coll_distr_anch" in hist
    assert tables["loading"][1] == [(1, 1), (1, 1)] and tables["loading_b"][1] == [(1, 1), (1, 1)]
    assert all(np.all(np.isfinite(h.y)) for h in hist.values())


# ---- bottlebrush ---------------------------------------------------------

BB = fn.Params(architecture="bottlebrush", nb=9, ns=3, step=3, lbox=LBOX, eps=0.1,
               ncoll1=1, ncoll2=2)


def bb_frame_array(rng, wrap=False):
    """Random bottlebrush frame: nb=9 (anchors 2, 5, 8), 3 side chains of 3, 1 + 2 colloids.

    Backbone along y; side chains radial; ids shuffled to check that order is irrelevant.
    """
    rows = []
    for j in range(1, 10):
        rows.append((j, 2, 50.0, 45.0 + j, 50.0))
    next_id = 10
    for anchor in (2, 5, 8):
        angle = rng.uniform(0, 2 * np.pi)
        for k in range(1, 4):
            rows.append((next_id, 1, 50 - k * np.sin(angle), 45.0 + anchor, 50 + k * np.cos(angle)))
            next_id += 1
    rows.append((19, 3, 55.0, 50.0, 50.0))
    rows.append((20, 4, 50.0, 50.0, 54.0))
    rows.append((21, 4, 45.0, 46.0, 48.0))
    arr = np.array(rows, float)
    if wrap:   # shift the whole system across the periodic boundary in y
        arr[:, 3] = (arr[:, 3] + 45.0) % LBOX
    return arr[rng.permutation(len(arr))]


def reference(frame, layout):
    """Direct (loop) port of the original bottlebrush analysis, for one frame."""
    pos = dict((int(r[0]), r[2:5]) for r in frame)
    types = dict((int(r[0]), int(r[1])) for r in frame)
    nb = layout.nb
    par, ort = [], []
    for anchor, arm in layout.side.items():
        vbone = pos[anchor + 1] - pos[anchor] if anchor != nb else pos[anchor] - pos[anchor - 1]
        for mon in arm:
            varm = pos[mon] - pos[anchor]
            p = np.dot(varm, vbone) / np.dot(vbone, vbone) * vbone
            par.append(np.linalg.norm(p))
            ort.append(np.linalg.norm(varm - p))
    brush = np.array([pos[i] for i in pos if types[i] in (1, 2)])
    cm = brush.mean(axis=0)
    centres = [np.mean([pos[i] for i in g], axis=0) for g in layout.groups]
    out = {"par": par, "ort": ort}
    for species, t in (("a", 3), ("b", 4)):
        cs = [pos[i] for i in pos if types[i] == t]
        out[species] = {
            "cyl": [min(np.linalg.norm(c - pos[i]) for i in range(1, nb + 1)) for c in cs],
            "eff": [min(np.linalg.norm(c - m) for m in centres) for c in cs],
            "cm": [np.linalg.norm(c - cm) for c in cs]}
    return out


@pytest.mark.parametrize("wrap", [False, True])
def test_bbrush_frame_matches_loop_reference(wrap):
    """The vectorised analysis equals a plain-loop port of the original algorithm."""
    layout = pp.BrushLayout(BB)
    assert layout.nb == 9 and layout.anchors == [2, 5, 8] and layout.blob_size == 1
    assert len(layout.groups) == 9
    frame = bb_frame_array(np.random.RandomState(4), wrap=False)
    if wrap:
        # same configuration, but wrapped: distances must not change
        moved = frame.copy()
        moved[:, 3] = (moved[:, 3] + 60.0) % LBOX
        frame_for_ref, frame = frame, moved
    else:
        frame_for_ref = frame
    obs = pp.bbrush_frame(frame, LBOX, layout)
    ref = reference(frame_for_ref, layout)
    assert np.allclose(sorted(obs["par"]), sorted(ref["par"]))
    assert np.allclose(sorted(obs["ort"]), sorted(ref["ort"]))
    for species in ("a", "b"):
        for kind in ("cyl", "eff", "cm"):
            assert np.allclose(sorted(obs[species][kind]), sorted(ref[species][kind]))
    assert obs["ree"] == pytest.approx(8.0)


def test_bbrush_analysis_writes_files(tmp_path):
    """Full analysis on a synthetic trajectory (first frame skipped); normalisations are finite."""
    rng = np.random.RandomState(0)
    os.makedirs(str(tmp_path / "configurations"))
    text = ""
    for step in range(4):
        arr = bb_frame_array(rng)
        lines = ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(arr)),
                 "ITEM: BOX BOUNDS pp pp pp"] + ["0.0 {}".format(LBOX)] * 3 + ["ITEM: ATOMS id type x y z"]
        lines += ["{} {} {} {} {}".format(int(r[0]), int(r[1]), *r[2:]) for r in arr]
        text += "\n".join(lines) + "\n"
    (tmp_path / "configurations" / "bbrush_0.lammpstrj").write_text(text)
    hist, tables = pp.analyse(str(tmp_path / "configurations"), BB)
    assert hist == {}
    assert sorted(tables) == ["coll_dist", "coll_dist_b", "coll_dist_nnorm", "coll_dist_nnorm_b",
                              "density_profile", "end_to_end"]
    assert len(tables["end_to_end"][1]) == 3
    assert tables["coll_dist_nnorm"][1][0][-1] == 3          # counter column
    out = tmp_path / "analysis"
    pp.write_results(hist, tables, str(out))
    lines = (out / "coll_dist.txt").read_text().splitlines()
    assert lines[0] == "# x    y_cyl   y_eff   y_cm"
    assert all(np.isfinite([float(v) for v in l.split()]).all() for l in lines[1:])
    # y_eff and y_cyl are different columns now (the old script wrote y_cyl twice)
    assert any(l.split()[1] != l.split()[2] for l in lines[1:] if float(l.split()[1]) > 0)


def test_bbrush_analysis_without_colloids(tmp_path):
    """colloids 0: only the density profile and end-to-end files."""
    params = BB.replace(colloids=0)
    os.makedirs(str(tmp_path / "configurations"))
    arr = bb_frame_array(np.random.RandomState(1))
    arr = arr[arr[:, 1] <= 2]
    lines = ["ITEM: TIMESTEP", "0", "ITEM: NUMBER OF ATOMS", str(len(arr)),
             "ITEM: BOX BOUNDS pp pp pp"] + ["0.0 {}".format(LBOX)] * 3 + ["ITEM: ATOMS id type x y z"]
    lines += ["{} {} {} {} {}".format(int(r[0]), int(r[1]), *r[2:]) for r in arr]
    frame = "\n".join(lines) + "\n"
    (tmp_path / "configurations" / "bbrush_0.lammpstrj").write_text(frame * 2)
    _, tables = pp.analyse(str(tmp_path / "configurations"), params)
    assert sorted(tables) == ["density_profile", "end_to_end"]
