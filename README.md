# polymer-colloid-pipeline

Scripts to set up, run and analyse molecular-dynamics simulations of a **polymer
interacting with colloids** in a periodic box, using
[LAMMPS](https://www.lammps.org), either on your own machine or on a PBS/Torque
cluster. Two architectures are supported, chosen with the `architecture` key of
`par.txt`:

| `architecture` | polymer | colloids (optional, `colloids 1/0`) |
|---|---|---|
| `star` | `f` arms of `ns` monomers on a core bead | one colloid of diameter `2 rs / q` |
| `bottlebrush` | backbone of `nb` beads with a side chain of `ns` beads every `step` beads | `ncoll1` of diameter `dcoll1` and `ncoll2` of diameter `dcoll2`, inserted at random |

Units are Lennard-Jones, temperature 1, Nosé–Hoover (NVT), FENE bonds.

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
| `colloids` | all | `1` with colloids (default), `0` bare polymer |
| `ns` | all | monomers per arm / per side chain |
| `sigma` | all | monomer diameter |
| `lbox` | all | edge of the cubic box |
| `restart` | all | `1` to continue from the restart file |
| `seed` | all | *(optional)* integer seed; omit for a random one |
| `eps` | with colloids | colloid–monomer attraction strength (bottlebrush default 0.1) |
| `f` | star | number of arms |
| `q` | star + colloid | star radius / colloid radius; star radius is `rs = 0.6 f^(1/5) ns^0.588` |
| `rcore` | star | *(optional, default 1.75)* core radius. Arms are placed by random sequential insertion: 1.75 fits at most ~42 arms, `f = 50` needs `>= 2.0`, `f = 60` `>= 2.25`; otherwise `generate_system.py` stops with an error |
| `nb`, `step` | bottlebrush | backbone length and grafting spacing |
| `ncoll1`, `dcoll1`, `ncoll2`, `dcoll2` | bottlebrush + colloids | number and diameter of the two colloid species (defaults 40, 14, 40, 10) |

Old `par.txt` files (without `architecture`) are read as star runs with a colloid.
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

* **star**: `mon_distr_anch`, `mon_distr_cm`, `rg`, and with a colloid `coll_distr_anch`,
  `coll_distr_anch_2rg`, `coll_distr_cm`, `loading.txt` (colloid within `rs` / `2 rs` per frame).
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
python3 -m pytest -q
```

`tests/data/in.star.expected` and `tests/data/in.bbrush.expected` are input files
written by the original scripts; the suite checks that the generators reproduce them
(the bottlebrush one except for the repulsive cutoff, which is now exactly 2^(1/6) instead of the rounded 1.1224).
