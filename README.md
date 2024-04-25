# SCOMAP-XD
## Scattering Contrast Match Points with Explicit-Atom Deuteration

Python Scritpting Code to deuterate PDB files for contrast match point prediction for Small Angle Neutron Scattering experiments with SASSENA.
There are three main scripts for deuteration of a pdb file(deuterate.py), preparation of the xml input file for scattering calculations with SASSENA(writeSASSENAxml.py),
and finally the analysis of the contrast match points(AnalyzeSASSENA.py). 

See the published paper for more details: https://doi.org/10.1107/S2059798323002899. 

SASSENA: (https://github.com/benlabs/sassena)
DOI:(http://dx.doi.org/10.1016/j.cpc.2012.02.010)

### Installation:
The code is written in python3.10. The python environment for SCOMAPXD was preparted with Anaconda. The package list for the environment is "SCOMAPXD_AnacondaEnvironment.txt". 
