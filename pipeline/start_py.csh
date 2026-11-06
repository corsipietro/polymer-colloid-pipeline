#!/bin/tcsh
# Submit one simulation to the PBS queue (cluster mode).
# Run from inside a run directory (it contains par.txt and the generation scripts).
# The job runs run_simulation.py: build the data file and input, then call LAMMPS.

# ---- site-specific settings: edit for your cluster ----
set PYTHON = python3.4
set LAMMPS = /storage/local/exp_soft/local_sl7/lammps/lammps-22Aug18/build/lmp
set WALLTIME = 48:00:00
# -------------------------------------------------------

setenv RUN `pwd`

echo "#\!/bin/tcsh" > start.script
echo "#PBS -l walltime=$WALLTIME" >> start.script
echo "cd '$RUN'" >> start.script
echo "setenv LAMMPS_CMD '$LAMMPS'" >> start.script
echo "$PYTHON run_simulation.py" >> start.script

chmod u+x start.script

qsub -N polymer_sim -o Standard-Out -e Standard-Err ./start.script > jobid.dat
