#!/bin/bash

#Loop through all the sD2O and gD2O conditions
#Remove all the water and solvent 
#Call Pepsi sans using hModel=3 
#If you have muliple structures and deuteration files, you will have to loop over those as well

for fdeut in 0 20 40 70 85 100
do
    for d2osolv in 0 20 40 60 80 90 100
    do
        fsd2o=`echo "scale=2 ; $d2osolv / 100" | bc`
        sed '/SOD/d' ../../Template_PDBs/MBP_${d2osolv}-sD2O_${fdeut}-gD2O_NEx0001.pdb > template_prot_only/MBP_${d2osolv}-sD2O_${fdeut}-gD2O_explicitH.pdb
        sed -i '/CLA/d' template_prot_only/MBP_${d2osolv}-sD2O_${fdeut}-gD2O_explicitH.pdb
        sed -i '/HOH/d' template_prot_only/MBP_${d2osolv}-sD2O_${fdeut}-gD2O_explicitH.pdb
      
        ~/./Pepsi-SANS template_prot_only/MBP_${d2osolv}-sD2O_${fdeut}-gD2O_explicitH.pdb -o MBP_${d2osolv}-sD2O_${fdeut}-gD2O_explicitH.dat \
			 -n 50 -ms 0.6 -ns 100 --hyd --hModel 3 --d2o ${fsd2o} 
    done
done
