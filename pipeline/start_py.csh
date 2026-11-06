#!/bin/tcsh
# Submit one simulation to the PBS queue: build star.dat and in.star, then run LAMMPS.
# Run from inside a run directory (it contains par.txt and the generate_*.py scripts).

# ---- site-specific settings: edit for your cluster ----
set PYTHON = python3.4
set LAMMPS = /storage/local/exp_soft/local_sl7/lammps/lammps-22Aug18/build/lmp
set WALLTIME = 48:00:00
# -------------------------------------------------------

setenv RUN `pwd`

echo "#\!/bin/tcsh" > start.script
echo "#PBS -l walltime=$WALLTIME" >> start.script
echo "cd '$RUN'" >> start.script
echo "$PYTHON generate_star.py" >> start.script
echo "$PYTHON generate_input.py" >> start.script
echo "$LAMMPS < in.star" >> start.script

chmod u+x start.script

qsub -N star_ads -o Standard-Out -e Standard-Err ./start.script > jobid.dat
