# polymer-colloid-pipeline

[![tests](https://github.com/corsipietro/polymer-colloid-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/corsipietro/polymer-colloid-pipeline/actions/workflows/tests.yml)

Scripts to set up, run and analyse molecular-dynamics simulations of a **polymer
interacting with colloids** in a periodic box, using
[LAMMPS](https://www.lammps.org), either on your own machine or on a PBS/Torque
cluster. Two architectures are supported, chosen with the `architecture` key of
`par.txt`:

| `architecture` | polymer | colloids (optional: `colloids 0/1/2` species) |
|---|---|---|
| `star` | `f` arms of `ns` monomers on a core bead | species 1 (diameter `2 rs / q`) and/or species 2 (`2 rs / q2`), `ncoll1` / `ncoll2` of each (default 1) |
| `bottlebrush` | backbone of `nb` beads with a side chain of `ns` beads every `step` beads | species 1 and/or 2, `ncoll1` / `ncoll2` of each (default 40), sizes from `q`/`q2` or `dcoll1`/`dcoll2`, inserted at random |

Both architectures can run **alone** (`colloids 0`), **with one colloid species** (`colloids 1`:
species 1 only, atom type 3) or **with two** (`colloids 2`: types 3 and 4). `ncoll1 1` gives a single
colloid.

Units are Lennard-Jones, temperature 1, Nosé–Hoover (NVT), FENE bonds.

**The key control parameter is the size ratio `q`** between the polymer and the colloid:
the colloid diameter is `2 R / q`, with `R` the radius of the star (`rs = 0.6 f^(1/5) ns^0.588`)
or of the brush's "effective star" (`rc = 0.5 ns^0.588 feq^(1/5)`, `feq = (g ns)^0.588`,
`g` = side chains per backbone monomer). Large `q` means small colloids. For the bottlebrush
you can give `q` (species 1) / `q2` (species 2) or, alternatively, explicit diameters.

> The physical description is inferred from the code; adjust it to the wording of
> the paper/thesis this belongs to.

## Layout

```
pipeline/
  functions.py            shared helpers: par.txt I/O, restart numbering, run discovery
  generate_system.py      par.txt -> star.dat / bbrush.dat      (entry point)
    generate_star.py, generate_bottle_brush.py                  (builders)
  generate_input.py       par.txt -> in.star / in.bbrush
  run_simulation.py       generate + run LAMMPS (used locally and by the cluster job)
  post_processing.py      configurations/*.lammpstrj -> analysis/*.txt
  launcher.py             starts a run/analysis locally or on the cluster
  cluster_submit.py       build a parameter scan and start every run
  cluster_restart.py      continue the runs of a scan from their restart files
  post_cluster_submit.py  run post_processing.py for every run of a scan
  retrieve_data.py        gather one analysis file of all runs in one folder
  start_py.csh, post_py.csh   PBS scripts (site-specific settings at the top)
example/                  par_star.txt, par_bottlebrush.txt
tests/                    pytest suite
```

## `par.txt`

One `key value` pair per line.

| key | applies to | meaning |
|---|---|---|
| `architecture` | all | `star` or `bottlebrush` (default `star`) |
| `mode` | all | `cluster` (PBS queue, default) or `local` (this machine) |
| `colloids` | all | number of colloid species: `0` polymer alone, `1` species 1, `2` species 1 and 2 (default: `1` for the star, `2` for the bottlebrush) |
| `ncoll1`, `ncoll2` | with colloids | colloids of species 1 (type 3) / 2 (type 4) (defaults: 1 for the star, 40 for the bottlebrush) |
| `ns` | all | monomers per arm / per side chain |
| `sigma` | all | monomer diameter |
| `lbox` | all | edge of the cubic box |
| `restart` | all | `1` to continue from the restart file |
| `seed` | all | *(optional)* integer seed; omit for a random one |
| `eps`, `eps2` | with colloids | colloid–monomer attraction strength of species 1 / 2 (`eps2` defaults to `eps`; bottlebrush `eps` default 0.1) |
| `f` | star | number of arms |
| `q`, `q2` | star + colloids | size ratio star radius / colloid radius of species 1 / 2 (`q2` required with `colloids 2`) |
| `rcore` | star | *(optional, default 1.75)* core radius. Arms are placed by random sequential insertion: 1.75 fits at most ~42 arms, `f = 50` needs `>= 2.0`, `f = 60` `>= 2.25`; otherwise `generate_system.py` stops with an error |
| `nb`, `step` | bottlebrush | backbone length and grafting spacing |
| `q`, `q2` | bottlebrush + colloids | size ratio effective-star radius / colloid radius of species 1, 2 (replaces the diameter) |
| `dcoll1`, `dcoll2` | bottlebrush + colloids | explicit diameters, used when `q` / `q2` is absent (defaults 14, 10) |

Old `par.txt` files (without `architecture`) are read as star runs with one colloid. Note that
for the bottlebrush `colloids 1` now means *one species*; the former behaviour (two species) is `colloids 2`.
Star colloids are placed at random, not overlapping each other and clear of the initial (straight-armed) star,
so the box must be large enough (an error asks to increase `lbox`).
The bottlebrush box must exceed `(nb + 2) sigma` along the backbone and
`2 (ns + 1) sigma` across it; the generator checks it.

## Running

Configure the scan in the PARAMETERS block of `pipeline/cluster_submit.py`
(architecture, mode, colloids, values to scan), then, from `pipeline/`:

```bash
python3 cluster_submit.py --dry-run     # create the directories and par.txt, start nothing
python3 cluster_submit.py               # start every run, in the mode set in the file
python3 cluster_submit.py --mode local  # override the mode
```

Each run directory gets its own `par.txt` and a copy of the scripts; it builds the
data file and the LAMMPS input, then runs LAMMPS (`run 1000000000`: it is stopped
by hand or by the wall time). Restart files are written periodically.

### Cluster mode

`start_py.csh` writes a PBS job and submits it with `qsub`. Edit the top of
`start_py.csh` and `post_py.csh` for your site (Python executable, LAMMPS path,
wall time). All runs of the scan are submitted at once.

### Local mode

The simulation is started in the background in the run directory
(`python3 run_simulation.py`, output in `run.log`). LAMMPS is called as
`$LAMMPS_CMD -in in.star` (or `in.bbrush`); `LAMMPS_CMD` defaults to `lmp` and may
include a launcher:

```bash
export LAMMPS_CMD="mpirun -np 4 lmp"
```

**In local mode, run one simulation at a time.** The simulations are open-ended and
all of them compete for the same cores, so several at once are slow; `cluster_submit.py`
refuses a local scan with more than one run unless `--allow-many` is given. Stop a
local run with `kill` on its process (find it with `ps`).

### Continue, analyse, collect

```bash
python3 cluster_restart.py <scan_dir> [--ns 100] [--mode local]   # next output index, nothing overwritten
python3 post_cluster_submit.py <scan_dir> [--ns 100]               # run the analysis of every run
python3 retrieve_data.py <scan_dir> --out data [--name rg]         # gather one analysis file
```

`mode` is taken from each run's `par.txt` unless `--mode` is given.

## Analysis output (`analysis/`)

Two-column histograms `x y` plus tables; the first frame of every trajectory file is skipped.

* **star**: `mon_distr_anch`, `mon_distr_cm`, `rg`, and for each colloid species present `coll_distr_anch`,
  `coll_distr_anch_2rg`, `coll_distr_cm`, `loading.txt` (number of colloids within `rs` / `2 rs` per frame);
  the files of species 2 end in `_b` (`coll_distr_anch_b`, `loading_b.txt`, ...).
* **bottlebrush**: `density_profile.txt` (side-chain atoms vs distance from their anchor,
  orthogonal and parallel to the local backbone), `coll_dist_nnorm.txt` and `coll_dist.txt`
  (colloid distance from the backbone, from the closest "effective star" and from the brush
  centre of mass; raw counts and normalised by cylindrical/spherical shells; `_b` files for the
  second colloid species) and `end_to_end.txt` (end-to-end distance and radii of gyration per frame).
  An effective star is a block of `n = round(0.5 ns (sigma ns)^(3/5) / (1 + sigma ns))`
  consecutive backbone atoms with their side chains.

LAMMPS also writes `gyration/gyr_all_<n>.txt` (and `gyr_bbone_<n>.txt`,
`bbone_extremes_<n>.lammpstrj` for the bottlebrush).

## Visualising a simulation

The scripts only produce numbers; to look at the system you need a molecular
visualiser, e.g. [VMD](https://www.ks.uiuc.edu/Research/vmd/) (used by the author).
Trajectories are in `configurations/<star|bbrush>_<n>.lammpstrj`:

```bash
vmd -lammpstrj configurations/star_0.lammpstrj
```

Atom types are 1 = monomer (side chain), 2 = core / backbone, 3 and 4 = colloids.
VMD draws all atoms with the same size by default: set the radius per type
(e.g. in *Graphics → Representations → VDW*, selection `type 3`) to see the colloid diameters.

## License

BSD 3-Clause, see `LICENSE`.

## Requirements and tests

* Python ≥ 3.4 (the run scripts avoid f-strings and annotations on purpose), `numpy`.
* LAMMPS with the `MOLECULE` package (`fene`, `lj/sf`).

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest -q                      # unit tests (LAMMPS tests are skipped without LAMMPS)
LAMMPS_CMD=/path/to/lmp python3 -m pytest -q   # also runs short real LAMMPS simulations
```

With LAMMPS available, `tests/test_lammps_smoke.py` runs 40000 steps of six variants
(each architecture alone, with one species and with two species), checks the
temperature, and runs the analysis. The radius of gyration computed by `post_processing.py`
agrees with LAMMPS' `compute gyration` frame by frame. Tested with LAMMPS 22 Jul 2025
(the cluster scripts were written for 22 Aug 2018).

`tests/data/in.star.expected` and `tests/data/in.bbrush.expected` are input files
written by the original scripts; the suite checks that the generators reproduce them
(the bottlebrush one except for the repulsive cutoff, which is now exactly 2^(1/6) instead of the rounded 1.1224).
