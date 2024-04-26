#!/bin/bash

cd $SLURM_SUBMIT_DIR

module purge
module load anaconda3 sassena
source activate ~/anaconda_SCOMAPXD/SCOMAPXD

## Deuterate the pdb step4_minim_vmd.pdb at solvent D2O conditions 0,20,40,60,80,90, and 100 % D2O at each growth condition 0,85
python3 ~/anaconda_SCOMAPXD/pyscripts/deuterate.py -s "0,20,40,60,80,90,100" -g "0,85" -n 20 -p "MBP" ./step4_minim_vmd.pdb

## SASSENA Calculation
for gd2o in 0 20 40 70 85 100
do
    for sd2o in 20 40 60 80 90 100
    do
        for nex in {01..20}
        do
        FNAME="MBP-RandHDX_${sd2o}-sD2O_${gd2o}-gD2O_NEx00${nex}.pdb"
        sed "s/XXTEMPFNAMEXX/${FNAME}/g" ./SASSENA_FACTORCalc.xml > ./SASSENA_FACTORCalc_${sd2o}-sD2O.xml
        sassena --config SASSENA_FACTORCalc_${sd2o}-sD2O.xml &> SASSENA_FactorCapture_${sd2o}-sD2O.out &
        sleep 5
        killall sassena
        TEMPF=$(grep "Target produces a background scattering length density" "SASSENA_FactorCapture_${sd2o}-sD2O.out" | grep -Eo '[+-]?[0-9]+([.][0-9]*)?')
        FACTOR=$(echo $TEMPF | awk '{print $2}')
        echo $FACTOR
        echo $sd2o $FACTOR >> sD2O_BCKGRNDFactor_Tseries.dat
        ## RUN SASSENA SANS Calculation
        sed "s/XXTEMPFNAMEXX/${FNAME}/g" SASSENA_SANSCalc_SoluteOnly_Template.xml > SASSENA_SANSCalc_SoluteOnly.xml
        sed -i "s/BCKGRNDFF/${FACTOR}/g" SASSENA_SANSCalc_SoluteOnly.xml
        mpirun sassena --config SASSENA_SANSCalc_SoluteOnly.xml
        mv signal.h5 SASSENA_RESULTS/MBP-RandHDX-Only_signal_${sd2o}-sD2O_${gd2o}-gD2O_Last_NEx00${nex}_HExF95_MolFracHDO_Kap1-52_Kp1.h5
        done
    done
done

## Calculate the solvent D2O matchpoints for all growth conditions
cd SASSENA_RESULTS/
python3 ~/anaconda_SCOMAPXD/pyscripts/AnalyzeSASSENA.py -g "0,20,40,70,85,100" -s "20,40,60,80,90,100" -m "sD2O" -p "MBP-RandHDX" -n 20 >> ../matchpoint_MBPRandHex.out

