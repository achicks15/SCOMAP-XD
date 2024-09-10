#!/bin/bash

#SBATCH -A m4748_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=100
#SBATCH -c 1
#SBATCH --gpus-per-node=4

#cd $SLURM_SUBMIT_DIR

module load python
conda activate ~/anaconda_SCOMAPXD/SCOMAPXD
source /global/cfs/projectdirs/m4748/sassena_7-29-2024/bin/SASSENARC.bash

PYTHONDIR=/global/homes/a/ach15/anaconda_SCOMAPXD/SCOMAPXD/bin
SCOMAPXDDIR=/global/homes/a/ach15/anaconda_SCOMAPXD/SCOMAP-XD

## Deuterate the pdb step4_minim_vmd.pdb at solvent D2O conditions 0,20,40,60,80,90, and 100 % D2O at each growth condition 0,85
${PYTHONDIR}/python3 $SCOMAPXDDIR/deuterate.py -s "${sd2o}" -g "0" -n 20 -p "XYLANASE-HDX_tip3phw" -t "step3_input.psf" ./step3_input.gro

## SASSENA Calculation
for gd2o in 0
do
        for nex in {01..20}
        do
        FNAME="Template_PDBs/XYLANASE-HDX_tip3phw_${sd2o}-sD2O_${gd2o}-gD2O_NEx00${nex}.pdb"
        ${PYTHONDIR}/python3 ${SCOMAPXDDIR}/writeSASSENAxml.py -f 31 -l 101 -s 1 --sel_name solute water --sel_index 0:2817 2818:84077 -nq 100 --min_q 0.0 --max_q 0.5 -res 1000 -b 0.0 -k 1.00 1.52 -o factor_${sd2o}.h5 -t water -x factor_${sd2o}.xml -i $FNAME -d step5_trjcat_nopbc.dcd
        sassena --config factor_${sd2o}.xml &> SASSENA_FactorCapture_${sd2o}-sD2O.out &
        sleep 30
        killall sassena
        rm factor_${sd2o}.h5
        TEMPF=$(grep "Target produces a background scattering length density" "SASSENA_FactorCapture_${sd2o}-sD2O.out" | grep -Eo '[+-]?[0-9]+([.][0-9]*)?')
        FACTOR=$(echo $TEMPF | awk '{print $2}')
        echo $FACTOR
        echo $sd2o $FACTOR >> sD2O_BCKGRNDFactor_Tseries.dat
        ## RUN SASSENA SANS Calculation
        OUTNAME=SASSENA_RESULTS/XYLANASE-HDX_tip3phw-System_signal_${sd2o}-sD2O_${gd2o}-gD2O_Last_NEx00${nex}_HExF95_MolFracHDO_Kap1-52_Kp1.h5
        ${PYTHONDIR}/python3 ${SCOMAPXDDIR}/writeSASSENAxml.py -f 31 -l 101 -s 1 --sel_name solute water --sel_index 0:2817 2818:84077 -nq 100 --min_q 0.0 --max_q 0.5 -res 1000 -b ${FACTOR} -k 1.00 1.52 -o ${OUTNAME} -t system -x SASSENA_SANSCalc_${sd2o}.xml -i $FNAME -d step5_trjcat_nopbc.dcd
        srun -N 1 --ntasks-per-node=100 sassena --config SASSENA_SANSCalc_${sd2o}.xml
        done
done

## Calculate the solvent D2O matchpoints for all growth conditions
#cd SASSENA_RESULTS/
#/global/homes/a/ach15/anaconda_SCOMAPXD/SCOMAPXD/bin/python3 ${SCOMAPXDDIR}/AnalyzeSASSENA.py -g "0" -s "0,10,20,30,40,50,60,70,90,100" -m "sD2O" -p "E140Cor30" -n 1 >> ../matchpoint_E140Cor30_gD2O71.out

