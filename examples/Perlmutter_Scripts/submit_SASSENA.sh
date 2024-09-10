#!/bin/bash

for sD2O in 0 8 20 30 40 50 60 80 90 100
do
	sbatch --export=ALL,sd2o=$sD2O SASSENA_Perlmutter_SCOMAPXD.sh
done
