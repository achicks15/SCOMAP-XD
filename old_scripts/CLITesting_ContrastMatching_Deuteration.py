#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:25:53 2021

@author: Alan Hicks, PhD
"""

import sys
import os
import argparse
import itertools
import csv
import numpy as np
import pandas as pd
import mdtraj as md

from scipy.optimize import fsolve
from Bio.PDB import *

def _convert_GD20_TDeut(pgrowth_d2o, psolv_d2o):
    """
    Determination of total deuteration not just non-exchangeable hydrogens!
    Function fron 1986 Lederer et.al. 
    g = c1*d2 + c2*d1 + c3*d1^2
    g = total fraction incorporated deuterons
    d2 = % D2O in dialysis buffer --> what will be used in solvent !!! This has no effect on the C-H bonded non-exchangeable hydrogens !!!
    d1 = % D2O in growth media
    C1,2,3 are constants 
    """
    ## How to deal with selected labeling schemes -- Occupancy within the pdb
    C1 = 0.187
    C2 = 0.36
    C3 = 0.34
    return C1 * (psolv_d2o/100) + C2 * (pgrowth_d2o/100)  + C3 * np.power(pgrowth_d2o/100, 2) 

def _convert_GD20_NEXH(pgrowth_d2o):
    """
    Empirical Function that will be determined by HDX-MS data:
    Converts the growth conditions of the system to determine the deuteration 
    level of non-exchangeable hydrogens.
    Quadratic function based off Lederer 1986
    From Fit of GFP data: array([0.40595313, 0.46273678])
    """
    C1= 0.40595313
    C2= 0.46273678
    #C1 = 0.42
    #C2 = 0.44
    return C1 * (pgrowth_d2o/100) + C2 * np.power(pgrowth_d2o/100,2) 

def molfrac_d2o(p, sd2o):
    """
    --------------------------------
    Variables:
    p = parameters to solve
    sd2o = given solvent D2O percentage!
    ---------------------------------
    Function to solve for the fraction of H2O, HDO and D2O when % solvent D2O (sd2o) is added to a system
    Solves the set of equations using fsolve during the randomization step:
    Xd + Xhd/2 = sd2o/100
    Xh + Xhd/2 = 1 - sd2o/100
    Ke*Xd*Xh = Xhd^2
    """
    nxd, nxhd, nxh = p
    ke=3.76 ## experimentally defined equilibrium constant
    return (nxd + nxhd/2 - sd2o/100, nxh + nxhd/2 - (1-sd2o/100), ke*nxd*nxh-nxhd**2)

def random_sets(percentsd20, percentgd20):
    """
    Determines the number of randomization iterations to account for all the probable combinations of 
    deuteratable hydrogens
    """
    true_gd20 = _convert_GD20_NEXH(percentgd20)
    if (percentsd20 == 0) & (percentgd20 == 0):
        return 1
    else:
        return (1+(-np.power(percentsd20/100,2)-np.power(true_gd20/100,2)+(percentsd20/100)*(true_gd20/100)+1)*10).astype(int)

def calc_hbonds(traj):
    """
    Function to calculate the hydrogen bonds within the protein for the
    calculation of the protection factors.
    """
    protein_select = traj.topology.select('protein')
    protein_traj = md.Trajectory(traj.xyz[0, protein_select], traj.topology.subset(protein_select))
    hbonds_bh = md.baker_hubbard(protein_traj)
    
    #label = lambda hbond : '{} -- {}'.format(traj.topology.atom(hbond[0]), traj.topology.atom(hbond[2]))
    hbond_array = []
    for hbond in hbonds_bh:
        hbond_array.append([traj.topology.atom(hbond[0]), hbond[0],
                            traj.topology.atom(hbond[1]), hbond[1],
                            traj.topology.atom(hbond[2]), hbond[2]])
    
    nhb_counts = (pd.DataFrame(hbond_array)[0].astype(str) + ' - ' + pd.DataFrame(hbond_array)[2].astype(str)).value_counts()
    return hbond_array, nhb_counts

def calc_ncontacts(traj, hex_list, cutoff = 6.5):
    
    """
    function to calculate the number of heavy atom contacts around the HA donor for the hydrogen
    """
    protein_select = traj.topology.select('protein')
    protein_traj = md.Trajectory(traj.xyz[0, protein_select], traj.topology.subset(protein_select))
    
    ncontacts_list = np.zeros((len(hex_list),1))
    haselect = protein_traj.topology.select('not type H')
    for hxindex, hxatom in enumerate(hex_list):
        
        pair_list = list(itertools.product([hxatom], haselect))
        ncontacts = ((md.compute_distances(protein_traj, pair_list, periodic=False)*10)<cutoff).sum()
        ncontacts_list[hxindex] = ncontacts
        
    return ncontacts_list

def calc_pfactor_BVModel(nhb, nc):
    """
    Model from Best and Vendruscolo, Structure, 14, 97-106, 2006
    nhb : number of hydrogen bonds the heavy atom - H pair are involved in
    nc : number of heavy atom contacts w/in cutoff (calc_ncontacts) to the heavy atom of the exchanging hydrogen
    """
    return (2*nhb + 0.35*nc)

def _random_select_protein(pgrowthD20, psolvD20, XexH_df, NexH_df, nsets, hexfactor=0.95, segdeut=False):
    
    """
    -----------------------------------------------------------------------------------------------
    Description:
    Function that does the random selection for proteins. It weights exchangeable hydrogen selection by the P-factor.
    It does a random deuteration unless segdeut is true for the non-exchangeable hydrogens. 
    This function should be run for each protein species in the system.  
    -----------------------------------------------------------------------------------------------
    Variables:
    pgrowthD20: the D2O percentage used in the growth of the protein
    psolvD20: the solvent D2O of the experiment
    XexH_df: Exchangeable hydrogen dataframe
    NexH_df: Non-exchangeable hydrogen dataframe
    nsets: # of randomizations to perform
    hexfactor: Scales back the total of the exchangeable hydrogens 
                ~= to 95% percent due to 90% backbone N-H exchange and 100% of side chain exchange
    segdeut: Boolean that decides whether the NexH_df should be randomized or are pre-selected.
    -----------------------------------------------------------------------------------------------
    """
    ## reset the random seed
    np.random.seed()
    ## NonExchangeable Hydrogens Exchange
    ## growth conditions but no segmental deuteration
    if (pgrowthD20 > 0.0)&(not segdeut):
        ## if there are growth D2O conditions, select the non-exchangeable hydrogens
        #percent_nex = _convert_GD20_NEXH(pgrowthD20, psolvD20)
        percent_nex = _convert_GD20_NEXH(pgrowthD20)
        random_select_nex = [np.random.choice(NexH_df.index, size=int(NexH_df.shape[0]*(percent_nex)),
                                          replace=False) for n in range(nsets)]
        random_select_nex = np.sort(np.vstack(random_select_nex), axis=1)
    elif (pgrowthD20==0.0)&(segdeut):
        ## NexH_df['H2DConvert'] is already a boolean so should select out the proper
        ## indices that need to be deuterated.
        ## !! This may need to be changed in the future for each protein/segment? Is this necessary? 
        nexd_indx = NexH_df[NexH_df['H2DConvert'].values].index.values
        random_select_nex = np.vstack([list(nexd_indx)]*nsets)
        
    elif (pgrowthD20>0.0)&(segdeut):
        print('Adding non-exchangeable hydrogens to the protein. Is this useful/what you really want?')
        raise SystemExit('Functionality not added yet')
        
    else:
        ## if no growth conditions, no Non-Exchangeable hydrogens are in the structure, so create a boolean
        ## array of false values so no selections are returned in pdb writing step.
        ## This can also be used when doing explicit/segmental/selective deuteration
        ## set pgrowthD2O to zero.
        random_select_nex = np.vstack([[False]*NexH_df.shape[0] for n in range(nsets)])
    
    if (psolvD20 > 0.0) & (psolvD20 < 100.0):
        
        ## Exchangeable Hydrogens Weighting
        ## Weighted random select should also be a potential empirical model. Its too simplistic right now.
        weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values
        random_select_xex = [np.random.choice(XexH_df.index.values,
                                              size=(int(XexH_df.shape[0]*(psolvD20/100)*hexfactor)), 
                                              replace = False, p=weights/weights.sum()) for n in range(nsets)]
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
         
        
    elif (psolvD20 == 100.0)&(hexfactor < 1.0):
        #print('solvent D2O is {} and hexfactor is {}'.format(psolvD20,hexfactor))
        weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values
        random_select_xex = [np.random.choice(XexH_df.index.values,
                                              size=(int(XexH_df.shape[0]*(psolvD20/100)*hexfactor)), 
                                              replace = False, p=weights/weights.sum()) for n in range(nsets)]
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
        
    elif psolvD20 == 100.0:
        ## Select All of the Solvent and Exchangeable Hydrogens??
        random_select_xex = np.vstack([[True]*XexH_df.shape[0] for n in range(nsets)])
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
        
    else:
        random_select_xex = np.vstack([[False]*XexH_df.shape[0] for n in range(nsets)])
    
    return random_select_xex, random_select_nex

def _random_select_solvent(psolvD20, SolvExH_df, nsets):
    
    """
    --------------------------------------------------------------------------
    Description:
    Function that does the random selection for the solvent molecules: Currently only works for H2O 
    Selects the indices for selection based off the proper molar fraction of H2O/HDO/D2O in solution at 
    the fraction of psolvD20.
    --------------------------------------------------------------------------
    Variables:
    psolvD20: experimental solvent D2O conditions
    SolvExH_df: solvent hydrogens dataframe
    nsets: number of randomization to choose
    --------------------------------------------------------------------------
    """
    
    np.random.seed()
    ## Determine the mixing fractions of the H2O-HDO-D2O in the solution
    ## H2O + D2O <-> 2HDO
    ## nxd = mol fraction of D2O
    ## nxhd = mol fraction of HDO
    ## nxh = mol fraction of H2O
    nxd, nxhd, nxh = fsolve(molfrac_d2o, (0.25, 0.5, 0.25), args=psolvD20)
    
    ## Integer math ....
    nxdhalf_size = np.floor(SolvExH_df.shape[0]*nxd/2).astype(int)
    nxhd_size = np.ceil(SolvExH_df.shape[0]/2*nxhd).astype(int)
    nxd_size = nxdhalf_size*2
    #if ((nxdhalf_size + nxhd_size) % 2) == 1:
    #    nxd_size += 1
    
    if (psolvD20 > 0.0) & (psolvD20 < 100.0):
         
        ##Solvent Selection 
        random_select_solvex = np.zeros((nsets, nxd_size + nxhd_size)).astype(int)
        #print(random_select_solvex.shape)
        for n in range(nsets):
            ## Every other index is a new water molecule in SolvExH_df
            ## Randomly select from this list of indices to select the H2O molecules for replacement
            d2o_indx = np.random.choice(SolvExH_df.index.values[::2],
                                        size=nxdhalf_size, replace=False)
            ## Remove the d2o_indx from the array of molecule indices and randomly choose from those for the 
            ## HDO molecules
            hdo_indx = np.random.choice(np.delete(SolvExH_df.index.values[::2], (d2o_indx/2).astype(int)),
                                        size=nxhd_size, replace=False)
            #print(hdo_indx.shape,d2o_indx.shape)
            d2o_indx = np.concatenate([d2o_indx, d2o_indx+1])
            #print(d2o_indx.shape)
            random_select_solvex[n,:] = np.sort(np.concatenate([d2o_indx, hdo_indx])).astype(int)
            
        #solvsize = int(SolvExH_df.shape[0]*(psolvD20/100))
            
        #random_select_solvex = [np.random.choice(SolvExH_df.index, size=solvsize,
        #                                         replace=False) for n in range(nsets)]
        #random_select_solvex = np.sort(np.vstack(random_select_solvex),axis=1)
        
        
    elif psolvD20 == 100.0:
        ## Select All of the Solvent and Exchangeable Hydrogens??
        solvsize = int(SolvExH_df.shape[0])
        random_select_solvex = [[True]*solvsize for n in range(nsets)]
        random_select_solvex = np.vstack(random_select_solvex)
        
    else:
        random_select_solvex = np.vstack([[False]*SolvExH_df.shape[0] for n in range(nsets)])
    
    return random_select_solvex

def write_pdb(filename, top_df) -> None:
    """ 
    function to write out a pdb file formatted from a dataframe
    Its faster to use the pandas to_csv function, so we can convert the pdb styled dataframe (top_df) 
    to a string for writing using a lambda function.
    
    """
    
    with open(filename, 'w') as pdbfilew:
        pdbfilew.write('TITLE \t TEMPLATE DEUTERATED PROTEIN\n')
        #pdbfile.write('CRYST1 \t {:}\t{}\t{}\t{}\t{}'.format())
        pdbfilew.write('MODEL \t 1\n')
        
        ## apply lambda function to the whole dataframe and write all at once 
        (top_df.apply(lambda line: "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}".format('ATOM ', int(line['serial']),
                                                                                                                                               line['name'], ' ', str(line['resName']), ' ',
                                                                                                                                               line['resSeq'], ' ', 
                                                                                                                                               line['x'], line['y'], line['z'],
                                                                                                                                               line['H2DConvert'], line['PFactor'],
                                                                                                                                               str(line.element), ' '), axis=1).to_csv(pdbfilew, index=False, header=False, sep='\t', quoting=csv.QUOTE_NONE))

        
        
        ### Old way iterating over every row:
        #for indx, line in ex_topdf.iterrows():
        #    stringdata = ['ATOM', int(line['serial']), line['name'], ' ', str(line['resName']), ' ', line['resSeq'], 
        #              ' ', line['x'], line['y'], line['z'], line['H2DConvert'],
        #              line['PFactor'], str(line.element), ' ']
        #    pf_string="{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(*stringdata)
        #    pdbfile.write(pf_string)
        pdbfilew.write('TER\n')
        pdbfilew.write('ENDMDL\n')
        pdbfilew.write('END\n')
    
    return None

def _read_segmentaldeut_pdb(pdbfname):
    """
    --------------------------------------------------------------------------
    Description:
    Function to read in the pdb to get the occupation numbers for segmental/specific deuteration.
    This PDB should have the deuteration of the solute ordered absolutely 
    the same way as the full solvated pdb.
    This function uses the BioPython PDBParser to read in the PDB.
    --------------------------------------------------------------------------
    Variables:
    pdbfname(str): Name of the pdb 
    --------------------------------------------------------------------------
    """
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('MBP_PerDeut', pdbfname)
    occupancy_list = [[indx, atom.occupancy] for indx, atom in enumerate(structure.get_atoms())]
    
    return occupancy_list



def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(usage = USAGE, 
                                      description=DESCRIPTION
                                      )
    parser.add_argument("-v","--version",action='version',
                        version=VERSION.format(parser.prog)
                        )
    parser.add_argument("-g","--percent_growthD2O",
                        type=str, default="0.0,85.0",
                        help="percent D2O present in the growth solvent media"
                        )
    parser.add_argument("--fdeuterated",
                        help="predicted fraction of deuteration from experiment")
    parser.add_argument("-s","--percent_solventD2O",
                        type=str, default="0.0,20.0,40.0,60.0,80.0,90.0,100.0",
                        help="percent D2O present in the experimental solvent media"
                        )
    parser.add_argument("-d","--segmental_deuteration",
                        default=False, action='store_true',
                        help="Boolean type to turn on segmental/specific deuteration."
                        )
    parser.add_argument("-p","--prefix",
                        type=str, 
                        help="prefix for naming the template pdbs")
    parser.add_argument("-n","--nrand_pdb",
                        type=int,
                        help="Number of randomized template PDBs to write for all conditions"
                        )
    parser.add_argument("pdbfile", nargs='*',
                        help="solvated pdb file with the system of interest")
    
    return parser

### GLOBAL Variable descriptions
USAGE="{} [options] [inputpdb]".format(sys.argv[0])
DESCRIPTION="Deuterate a system of interest for contrast matching prediction"
VERSION="{} version 0.0.1"
####

if __name__=='__main__':
    
    mainparser= init_argparse()
    mainargs = mainparser.parse_args()
    #print(mainargs.pdbfile)
    if (len(mainargs.pdbfile) == 1) & (not mainargs.segmental_deuteration):
        
        print("Segmental Deuteration is Off.")
        print("One PDB file detected as {} for deuteration".format(mainargs.pdbfile[0]))
        
    elif (len(mainargs.pdbfile) == 1) & (mainargs.segmental_deuteration):
        
        print("Segmental Deuteration is On.")
        print(mainargs.pdbfile)
        raise SystemExit("Two PDBs should be provided in order: pdbfile segdeutpdb")
        
    elif (len(mainargs.pdbfile) == 2) & (mainargs.segmental_deuteration):
        
        print("Segmental Deuteration is On.")
        print("One PDB file detected as {} for deuteration".format(mainargs.pdbfile[0]))
        print("Using PDB file {} as template for non-exchangeable hydrogen deuteration".format(mainargs.pdbfile[1]))
        
    else:
        raise SystemExit("No PDB provided for deuteration. Use -h or --help for more details.")
        
    ## Parse PDB:
        ## Determine species/number of molecules
        ## This will determine how the growth and solvent conditions are parsed eventually...
    PDB = md.load(mainargs.pdbfile[0])
    topologydf, bonds = PDB.topology.to_dataframe()
    coorddf = pd.DataFrame(PDB.xyz[0,:]*10, columns=['x','y','z'])
    bfactordf = pd.DataFrame(index=topologydf.index).fillna(0.0)
    pdbdf = topologydf.merge(coorddf, left_index=True, right_index=True)
    
    ## Topology Manipulation -- Protein
    prot_hydrogens = PDB.topology.select('protein and type H')
    prot_OorN = PDB.topology.select('protein and (type N or type O)')
    ## Preparing Arrays for the exchangeable and non-exchangeable hydrogens
    exchangeable_hydrogens = []
    nonexchangeable_hydrogens = []

    ## Bond Selection
    ## determine if the Hydrogen is bonded to a carbon (non-exchangeable hydrogen)
    ## or nitrogen/oxygen/sulfur (exchangeable hydrogen)
    for bond in PDB.topology.subset(PDB.topology.select('protein')).bonds:
        ba1,ba2 = bond[0], bond[1]
        ele1,ele2 = list(PDB.topology.atom(ba1.index).element)[2], list(PDB.topology.atom(ba2.index).element)[2]
        atype1, atype2 = str(ba1).split('-')[1], str(ba2).split('-')[1]
        a1isNorO = ('N' in ele1) or ('O' in ele1) or ('S' in ele1)
        a1isC = ('C' in ele1)
        a2isH = ('H' in ele2) ## Eta is H in Greek! Can't use the naming type
        if (a1isNorO) and a2isH:
            exchangeable_hydrogens.append([ba1, ba1.serial-1, ba2, ba2.serial-1])
        elif a1isC and a2isH:
            nonexchangeable_hydrogens.append([bond[0], bond[0].serial-1, bond[1], bond[1].serial-1])
    
    ExHDF = pd.DataFrame(exchangeable_hydrogens,columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    ExHDF['N_HydBonds'] = 0
    ExHDF['N_Contacts'] = 0
    NExHDF = pd.DataFrame(nonexchangeable_hydrogens,columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    NExHDF['H2DConvert'] = False 

    if mainargs.segmental_deuteration:
         ## segmental PDB test
         segdeut_pdb = mainargs.pdbfile[1]
         segNEX_Occ = _read_segmentaldeut_pdb(segdeut_pdb)
         NExHDF_HI = NExHDF.set_index('H_Index')
         NExHDF_HI.loc[np.where(np.array(segNEX_Occ)[:,1]==1.0)[0],'H2DConvert'] = True
         NExHDF = NExHDF_HI.reset_index()[NExHDF.columns]
    
    ## Structure Calculations and Adding the PFactors
    hbond_array, NHBCounts = calc_hbonds(PDB)
    ncontacts_pdb = calc_ncontacts(PDB, ExHDF['HA_Donor_Index'].values)

    ExHDF['BondPair'] = ExHDF['HA_Donor'].astype('str') + ' - '  + ExHDF['ExH'].astype('str')
    ExHDF = ExHDF.set_index('BondPair')
    ExHDF.loc[NHBCounts.index, 'N_HydBonds'] = NHBCounts.values
    ExHDF = ExHDF.reset_index()
    ExHDF['N_Contacts'] = ncontacts_pdb.flatten().astype(int)
    ExHDF['PFactor'] = ExHDF.apply(lambda rr: calc_pfactor_BVModel(rr.N_HydBonds, rr.N_Contacts), axis=1)
    ## Sort by P-factor in ascending order: Lower P-factors means more likely to exchange
    sorted_index = ExHDF.sort_values('PFactor', ascending=True).index.values

    ## Solvent Selection and Manipulation
    solvent_H = PDB.topology.select('(resname HOH or resname TP3 or resname WAT) and type H')
    solvent_O = PDB.topology.select('(resname HOH or resname TP3 or resname WAT) and type O')
    SolventHDF = pd.DataFrame(index=np.arange(len(solvent_H)),columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    SolventHDF['H2DConvert'] = False 
    SolventHDF['H_Index'] = solvent_H
    SolventHDF['ExH'] = SolventHDF['H_Index'].apply(lambda solvh: PDB.topology.atom(solvh))
    SolventHDF['HA_Donor_Index'] = np.vstack([solvent_O.T,solvent_O.T]).T.flatten()
    SolventHDF['HA_Donor'] = SolventHDF['HA_Donor_Index'].apply(lambda solvo: PDB.topology.atom(solvo))
    
    SOD_indx = topologydf[topologydf['name']=='Na+'].index
    topologydf.loc[SOD_indx,'name'] = 'SOD'
    topologydf.loc[SOD_indx,'resName']='SOD'
    CLA_indx = topologydf[topologydf['name']=='Cl-'].index
    topologydf.loc[CLA_indx,'name'] = 'CLA'
    topologydf.loc[CLA_indx,'resName']='CLA'

    pgrowthD2O_list = [float(pg) for pg in mainargs.percent_growthD2O.split(',')]
    psolventD2O_list = [float(pg) for pg in mainargs.percent_solventD2O.split(',')]
    print(pgrowthD2O_list)
    print(psolventD2O_list)
    
    pdbprefixloc = '/'.join((mainargs.pdbfile[0].split('/')[:-1]))
    if mainargs.prefix:
        pbdsuffix=mainargs.prefix
    else:
        pbdsuffix = mainargs.pdbfile[0].split('/')[-1].split('.pdb')[0]
    print('Setting the pdb prefix to :{}'.format(pbdsuffix))
    temp_pdbdir = '{}/Template_PDBs/'.format(pdbprefixloc)
    try:
        os.mkdir(temp_pdbdir)
    except:
        print('directory already exists')
   # print(mainargs.nrand_pdb)
    ## Begin write loop
    for psD2O in psolventD2O_list:
        for pgD2O in pgrowthD2O_list:
            if mainargs.nrand_pdb:
                nrandom_sets=mainargs.nrand_pdb
            else:
                nrandom_sets = random_sets(psD2O, pgD2O)
            #print(nrandom_sets)
            ## Select only the top percentSD20 based on the p-factor
            D2HSorted = sorted_index[np.arange(0, int(np.floor((psD2O/100)*sorted_index.shape[0])),1)]
            ## Create the Pandas DataFrame column with boolean to determine if we're going to convert the name of the hydrogen  
            #nrandom_sets=20
            ExHDF['H2DConvert'] = False ## Always reset to False!! 
        
        
        ## Selecting the random indices for the exchange
            HXex_H2D_index, nex_h2d_index = _random_select_protein(pgD2O, psD2O, 
                                                                   ExHDF, 
                                                                   NExHDF, 
                                                                   nrandom_sets,
                                                                   hexfactor=1.0,
                                                                   segdeut=mainargs.segmental_deuteration
                                                                   )
        
            solvex_h2d_index = _random_select_solvent(psD2O, SolventHDF, nrandom_sets)
        
        
            for pdbnum, hdx_index, nex_index, solv_index in zip(range(nrandom_sets), HXex_H2D_index[:nrandom_sets],
                                                                nex_h2d_index[:nrandom_sets], solvex_h2d_index[:nrandom_sets]):
                ## Every iteration copy the dataframes:
                ex_topdf = topologydf.copy()
                ex_topdf = ex_topdf.merge(coorddf, how='left',left_index=True,right_index=True)
                ex_bfdf = bfactordf.copy() ## Holdes BFactors for writing the 
                ## No HDX at ALL
                #hdx_index = [False]*ExHDF.shape[0]
                ## Convert the names for:
                ### Exchangeable Hydrogens
                hex_h2ddf = ExHDF.copy()
                hex_h2ddf.loc[hdx_index, 'H2DConvert']=1
                ex_topdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values, 'name'] = 'D'
                ex_topdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values, 'element'] = 'D'
            
                ### NonExchangeable Hydrogens
                nex_h2df = NExHDF.copy()
                nex_h2df.loc[nex_index,'H2DConvert']=2
                ex_topdf.loc[nex_h2df.loc[nex_index,'H_Index'].values, 'name'] = 'D'
                ex_topdf.loc[nex_h2df.loc[nex_index, 'H_Index'].values, 'element'] = 'D'
            
                ### Solvent Hydrogens:
                solvex_h2df = SolventHDF.copy()
                #solv_index_paired = SolventHDF_Pairs.loc[solv_index, ['H1_Index','H2_Index']].values.flatten()
                solvex_h2df.loc[solv_index, 'H2DConvert'] = 3
            
                ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values,'name'] = 'D'
                ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'element'] = 'D'
            
                ## bfactor placement to make sure that the p-factor is added to the ExHDF
                ex_bfdf = (ex_bfdf.join(hex_h2ddf[['H_Index', 'PFactor', 'H2DConvert']]
                            .set_index('H_Index'), how='left').fillna(0.0))
    
                ex_bfdf.loc[nex_h2df.loc[nex_index, 'H_Index'].values, 'H2DConvert'] = 2
                ex_bfdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'H2DConvert'] = 3
    
                ex_bfdf['H2DConvert'] = ex_bfdf['H2DConvert'].astype(int)
    
                ex_topdf = ex_topdf.merge(ex_bfdf, how='left',left_index=True,right_index=True)
                
                if mainargs.segmental_deuteration:
                    Dpdb_fname = '{}/{}_{}-sD2O_SegDeut_NEx{:04}.pdb'.format(temp_pdbdir,pbdsuffix, int(psD2O), pdbnum+1)
                else:    
                    Dpdb_fname = '{}/{}_{}-sD2O_{}-gD2O_NEx{:04}.pdb'.format(temp_pdbdir,pbdsuffix, int(psD2O), int(pgD2O), pdbnum+1)
                write_pdb(Dpdb_fname, ex_topdf)
                #break;
    