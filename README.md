# star-colloid-pipeline

Scripts to set up, run and analyse molecular-dynamics simulations of **one star
polymer interacting with one colloid** in a periodic box, using
[LAMMPS](https://www.lammps.org) on a PBS/Torque cluster.

The star has `f` arms of `ns` monomers each, attached to a core bead. The colloid
is a single larger bead with a repulsive core and an attractive tail towards the
star beads. A scan varies the colloid/star size ratio `q` and the attraction
strength `eps` at a fixed, very low volume fraction (one star + one colloid in a
large box), and the analysis measures how the star shape (radius of gyration,
monomer density) and the colloid position (distance distribution, number of
colloids "loaded" inside the star) respond.

> The physical description above is inferred from the code; adjust it to the
> wording of the paper/thesis this belongs to.

## Layout

```
pipeline/
  functions.py          shared helpers (par.txt I/O, restart numbering, run discovery)
  generate_star.py      par.txt  ->  star.dat   (LAMMPS data file)
  generate_input.py     par.txt  ->  in.star    (LAMMPS input)
  post_processing.py    configurations/*.lammpstrj  ->  analysis/*.txt
  cluster_submit.py     build the scan directory tree and submit every run
  cluster_restart.py    continue the runs of a scan from their restart files
  post_cluster_submit.py  submit post_processing.py for every run of a scan
  retrieve_data.py      gather coll_distr_anch.txt of all runs in one folder
  start_py.csh, post_py.csh   PBS submission scripts (site-specific settings at the top)
example/par.txt         example run parameters
tests/                  pytest suite
```

## Workflow

1. **Configure** the scan at the top of `pipeline/cluster_submit.py`
   (`ARMS`, `NS`, `RATIO`, `EPS`, `PHI`, `RCORE`) and the cluster settings at the top of
   `start_py.csh` / `post_py.csh` (Python executable, LAMMPS path, wall time).
2. **Submit** (from the `pipeline/` directory):
   ```bash
   python3 cluster_submit.py --dry-run   # check the directory tree first
   python3 cluster_submit.py
   ```
   This creates `<date>_f<arms>/f_<f>/ns_<ns>/eps_<eps>/q_<q>/` for every
   combination, with its own `par.txt`. Each job runs `generate_star.py`,
   `generate_input.py`, then `lmp < in.star` (up to 48 h, restarts written every
   100000 steps, configurations every 500000 steps).
3. **Continue** unfinished runs (outputs get the next index, nothing is overwritten):
   ```bash
   python3 cluster_restart.py <scan_dir> [--ns 100]
   ```
4. **Analyse**:
   ```bash
   python3 post_cluster_submit.py <scan_dir> [--ns 100]
   python3 retrieve_data.py <scan_dir> --out data
   ```

## `par.txt`

| key | meaning |
|---|---|
| `f` | number of arms |
| `ns` | monomers per arm |
| `sigma` | monomer diameter (LJ units) |
| `lbox` | edge of the cubic box |
| `q` | star radius / colloid radius |
| `eps` | colloid–monomer attraction strength |
| `restart` | `1` to continue from `star2.restart` |
| `rcore` | *(optional, default 1.75)* core radius. Arms are placed on the core by random sequential insertion: at 1.75 at most ~42 arms fit, `f = 50` needs `rcore >= 2.0`, `f = 60` needs `>= 2.25`. If it does not fit, `generate_star.py` stops with an explicit error. |

The star radius used throughout is `rs = 0.6 f^(1/5) ns^0.588`.

## Output of `post_processing.py` (`analysis/`)

`mon_distr_anch`, `mon_distr_cm`, `coll_distr_anch`, `coll_distr_anch_2rg`,
`coll_distr_cm`, `rg` (two columns `x y`) and `loading.txt` (per frame: colloid
within `rs`, within `2 rs`). The first frame of every trajectory file is skipped.

## Requirements and tests

* Python ≥ 3.4 (the cluster scripts avoid f-strings and annotations on purpose), `numpy`.
* LAMMPS with the `MOLECULE` package (`fene`, `lj/sf`).

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest -q
```

`tests/data/in.star.expected` is an `in.star` produced by the original scripts;
the suite checks that the refactored generator reproduces it byte for byte.
