#!/bin/tcsh
# Submit the trajectory analysis of one run to the PBS queue.
# Run from inside a run directory (it contains post_processing.py).

# ---- site-specific settings: edit for your cluster ----
set PYTHON = python3.4
set WALLTIME = 5:00:00
# -------------------------------------------------------

setenv RUN `pwd`

echo "#\!/bin/tcsh" > post.script
echo "#PBS -l walltime=$WALLTIME" >> post.script
echo "cd '$RUN'" >> post.script
echo "$PYTHON post_processing.py" >> post.script

chmod u+x post.script

qsub -N ads_post_pr -o Standard-Post-Out -e Standard-Post-Err ./post.script > p_jobid.dat
