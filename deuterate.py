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

def calc_hbonds(selstr, solbool=False):
    """
    Function to calculate the hydrogen bonds within the protein for the calculation of the protection factors
    """
    protein_select = PDB.topology.select(selstr)
    protein_traj = PDB.atom_slice(protein_select)
    hbonds_bh = md.baker_hubbard(protein_traj)
    
    #label = lambda hbond : '{} -- {}'.format(PDB.topology.atom(hbond[0]), PDB.topology.atom(hbond[2]))
    hbond_array = []
    for hbond in hbonds_bh:
        hbond_array.append([protein_traj.topology.atom(hbond[0]), hbond[0],
                            protein_traj.topology.atom(hbond[1]), hbond[1],
                            protein_traj.topology.atom(hbond[2]), hbond[2]])
    
    if solbool:
        nhb_counts = (pd.DataFrame(hbond_array)[0].astype(str) 
                      + ' - ' + pd.DataFrame(hbond_array)[2].astype(str) 
                      + ' - ' + pd.DataFrame(hbond_array)[3].astype(str)
                     ).value_counts(sort=False)
    else:
        nhb_counts = (pd.DataFrame(hbond_array)[0].astype(str) 
                      + ' - ' + pd.DataFrame(hbond_array)[2].astype(str) 
                      #+ ' - ' + pd.DataFrame(hbond_array)[3].astype(str)
                     ).value_counts(sort=False)
    return hbond_array, nhb_counts

def calc_ncontacts(selstr, hex_list, cutoff = 6.5):
    
    """
    function to calculate the number of heavy atom contacts around the HA donor for the hydrogen
    """
    #protein_select = PDB.topology.select(selstr)
    #protein_traj = md.Trajectory(PDB.xyz[0, protein_select], PDB.topology.subset(protein_select))
    
    ncontacts_list = np.zeros((len(hex_list),1))
    haselect = PDB.topology.select('{} and not type H'.format(selstr))
    for hxindex, hxatom in enumerate(hex_list):
        
        pair_list = list(itertools.product([hxatom], haselect))
        ncontacts = ((md.compute_distances(PDB, pair_list, periodic=False)*10)<cutoff).sum()
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
    if (pgrowthD20 > 0.0)and(not segdeut):
        ## if there are growth D2O conditions, select the non-exchangeable hydrogens
        #percent_nex = _convert_GD20_NEXH(pgrowthD20, psolvD20)
        percent_nex = _convert_GD20_NEXH(pgrowthD20)
        random_select_nex = [np.random.choice(NexH_df.index, size=int(NexH_df.shape[0]*(percent_nex)),
                                          replace=False) for n in range(nsets)]
        random_select_nex = np.sort(np.vstack(random_select_nex), axis=1)
    elif (pgrowthD20==0.0) and  (segdeut):
        ## NexH_df['H2DConvert'] is already a boolean so should select out the proper
        ## indices that need to be deuterated.
        ## !! This may need to be changed in the future for each protein/segment? Is this necessary? 
        nexd_indx = NexH_df[NexH_df['H2DConvert'].values].index.values
        random_select_nex = np.vstack([list(nexd_indx)]*nsets)
        
    elif (pgrowthD20>0.0) and (segdeut):
        print('Adding non-exchangeable hydrogens to the protein. Is this useful/what you really want?')
        raise SystemExit('Functionality not added yet')
        
    else:
        ## if no growth conditions, no Non-Exchangeable hydrogens are in the structure, so create a boolean
        ## array of false values so no selections are returned in pdb writing step.
        ## This can also be used when doing explicit/segmental/selective deuteration
        ## set pgrowthD2O to zero.
        random_select_nex = np.vstack([[False]*NexH_df.shape[0] for n in range(nsets)])
    
    if (psolvD20 > 0.0) and (psolvD20 < 100.0):
        
        ## Exchangeable Hydrogens Weighting
        ## Weighted random select should also be a potential empirical model. Its too simplistic right now.
        if (not mainargs.solute_randhex):
            weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values    
        else:
            weights = np.array([1/len(XexH_df.index.values)]*(len(XexH_df.index.values)))
        
        random_select_xex = [np.random.choice(XexH_df.index.values,
                                              size=(int(XexH_df.shape[0]*(psolvD20/100)*hexfactor)), 
                                              replace = False, p=weights/weights.sum()) for n in range(nsets)]
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
         
        
    elif (psolvD20 == 100.0) and (hexfactor < 1.0):
        #print('solvent D2O is {} and hexfactor is {}'.format(psolvD20,hexfactor))
        if (not mainargs.solute_randhex):
            weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values
        else:
            weights = np.array([1/len(XexH_df.index.values)]*(len(XexH_df.index.values)))
        #weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values
        random_select_xex = [np.random.choice(XexH_df.index.values,
                                              size=(int(XexH_df.shape[0]*(psolvD20/100)*hexfactor)), 
                                              replace = False, p=weights/weights.sum()) for n in range(nsets)]
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
        
    elif (psolvD20 == 100.0)and(hexfactor == 1.0):
        ## Select All of the Solvent and Exchangeable Hydrogens??
        random_select_xex = np.vstack([[True]*int(XexH_df.shape[0]*hexfactor) for n in range(nsets)])
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
        
    else:
        random_select_xex = np.vstack([[False]*XexH_df.shape[0] for n in range(nsets)])
    
    return random_select_xex, random_select_nex

def _random_select_nucleic(pgrowthD20, psolvD20, XexH_df, NexH_df, nsets, hexfactor=1.0, segdeut=False):
    
    """
    -----------------------------------------------------------------------------------------------
    Description:
    Function that does the random selection for nucleic acids.
    It weights exchangeable hydrogen selection by the P-factor (Using the same weighting as proteins).
    This could be adjusted as necessary.
    No random deuteration of the non-exchangeable as there is no model for the incorporation of deuterium
    in nucleic acids during growth. Keeping segmental deuteration. Always set pgrowth=0.0.
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
    pgrowthD20=0.0
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
        print('Adding non-exchangeable hydrogens to the nucleic acids. Is this useful/what you really want?')
        print('Functionality not added yet')
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
        print('solvent D2O is {} and hexfactor is {}'.format(psolvD20,hexfactor))
        weights = XexH_df['PFactor'].apply(lambda pf : 1 - pf/XexH_df['PFactor'].max()).values
        random_select_xex = [np.random.choice(XexH_df.index.values,
                                              size=(int(XexH_df.shape[0]*(psolvD20/100)*hexfactor)), 
                                              replace = False, p=weights/weights.sum()) for n in range(nsets)]
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
    
    elif (psolvD20 == 100.0)&(hexfactor == 1.0):
        ## Select All of the Solvent and Exchangeable Hydrogens??
        random_select_xex = np.vstack([[True]*XexH_df.shape[0] for n in range(nsets)])
        random_select_xex = np.sort(np.vstack(random_select_xex), axis=1)
        
    else:
        random_select_xex = np.vstack([[False]*XexH_df.shape[0] for n in range(nsets)])
    
    return random_select_xex, random_select_nex



def _random_select_lipid(ldtypedf, psolvD20, XexH_df, NexH_df, nsets, hexfactor=1.0, segdeut=False):
    
    random_select_nex = np.zeros(nsets).reshape(nsets,1)
    random_select_xex = np.zeros(nsets).reshape(nsets,1)
    
    for ldresname in ldtypedf.index:
        
        ## Exchangeable Hydrogens
        if XexH_df.shape[0] > 0:
            hexdf_resdf = XexH_df[XexH_df['HA_Donor_resName']==ldresname].copy()
            if (psolvD20>0.0) & (psolvD20<100.0):
                reshex_rand = np.vstack([np.random.choice(hexdf_resdf.index, size=(int(hexdf_resdf.shape[0]*(psolvD20/100))), replace=False) for n in range(nsets)])
            
            elif psolvD20 == 100.0:
                random_select_xex = random_select_xex.astype(bool)
                reshex_rand = np.vstack([[True]*hexdf_resdf.shape[0] for n in range(nsets)])
            else:
                random_select_xex = random_select_xex.astype(bool)
                reshex_rand = np.vstack([[False]*hexdf_resdf.shape[0] for n in range(nsets)])
            
            random_select_xex = np.hstack([random_select_xex, reshex_rand])
        
    
        ## Non-exchangeable Hydrogens

        nexdf_resdf = NexH_df[NexH_df['HA_Donor_resName'] == ldresname].copy()
        if ('sterol' in ldtypedf.loc[ldresname,'acyl1_type'])&(ldtypedf.loc[ldresname,'acyl2']==None):
            fdeut = float(ldtypedf.loc[ldresname,'acyl1'])
            st_rand = np.vstack([np.random.choice(nexdf_resdf.index,size=int(nexdf_resdf.shape[0]*fdeut), replace=False) for n in range(nsets)])
            random_select_nex = np.hstack([random_select_nex,st_rand])
        else:
            print(ldresname)
            for lddomain in ['head','acyl2','acyl1']:
                ldindex = nexdf_resdf[nexdf_resdf['HA_Donor_Domain']==lddomain].index
                fdeut = float(ldtypedf.loc[ldresname,lddomain])
                #print(lddomain,fdeut)
                rand_lddom = np.vstack([np.random.choice(ldindex, size=int(ldindex.shape[0]*fdeut), replace=False) for n in range(nsets)])
                #print(rand_lddom.shape[1],ldindex.shape[0])
                #print(rand_lddom)
                random_select_nex = np.hstack([random_select_nex, rand_lddom])
                
    return np.sort(random_select_xex[:,1:]), np.sort(random_select_nex[:,1:])

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

def _random_select_solvent_RANDH2D(psolvD20, SolvExH_df, nsets):
    
    """
    Function that does the random selection
    This function weights the Exchangeable Hydrogens by the P-factor 
    How to convert to a class functions?
    """
    
    np.random.seed()
    ## Determine the mixing fractions of the H2O-HDO-D2O in the solution
    ## H2O + D2O <-> 2HDO
    ## nxd = mol fraction of D2O
    ## nxhd = mol fraction of HDO
    ## nxh = mol fraction of H2O
    nxd, nxhd, nxh = fsolve(molfrac_d2o, (0.25,0.5,0.25), args=psolvD20)
    
    ## Integer math ....
    #nxdhalf_size = np.floor(SolvExH_df.shape[0]*nxd/2).astype(int)
    #nxhd_size = np.ceil(SolvExH_df.shape[0]/2*nxhd).astype(int)
    #nxd_size = nxdhalf_size*2
    #if ((nxdhalf_size + nxhd_size) % 2) == 1:
    #    nxd_size += 1
    
    if (psolvD20 > 0.0) & (psolvD20 < 100.0):
         
        ##Solvent Selection 
        #random_select_solvex = np.zeros((nsets, nxd_size + nxhd_size)).astype(int)
        #print(random_select_solvex.shape)
        #for n in range(nsets):
            ## Every other index is a new water molecule in SolvExH_df
            ## Randomly select from this list of indices to select the H2O molecules for replacement
        #    d2o_indx = np.random.choice(SolvExH_df.index.values[::2],
        #                                size=nxdhalf_size, replace=False)
            ## Remove the d2o_indx from the array of molecule indices and randomly choose from those for the 
            ## HDO molecules
        #    hdo_indx = np.random.choice(np.delete(SolvExH_df.index.values[::2], (d2o_indx/2).astype(int)),
        #                                size=nxhd_size, replace=False)
            #print(hdo_indx.shape,d2o_indx.shape)
        #    d2o_indx = np.concatenate([d2o_indx, d2o_indx+1])
            #print(d2o_indx.shape)
        #    random_select_solvex[n,:] = np.sort(np.concatenate([d2o_indx, hdo_indx])).astype(int)
            
        solvsize = int(SolvExH_df.shape[0]*(psolvD20/100))
            
        random_select_solvex = [np.random.choice(SolvExH_df.index, size=solvsize,
                                                 replace=False) for n in range(nsets)]
        random_select_solvex = np.sort(np.vstack(random_select_solvex),axis=1)
        
        
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
    
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure('PDB_SD', pdbfname)
    print(structure[0])
    occupancy_list = [[indx, atom.occupancy] for indx, atom in enumerate(structure.get_atoms())]
    
    return occupancy_list

def _parse_topology(lpd_res='POPC')->dict:
    """
    function to parse the topology into sections and define protein systems, RNA, DNA and solvent (other than water)
    --------------------------------------------------------------------------
    Variables:
    
    Return:
    
    --------------------------------------------------------------------------
    """
    ## Mostly using CHARMM naming conventions for ease of use
    rnaselect = 'resname URA U' # only Uracil in RNA
    dnaselect = 'resname THY DT T' # only Thymine in DNA
    protselect = 'resname ARG ALA ASP ASN HIS HIE HID HIP HSD HSE LYS PRO GLY GLU GLN TRP TYR SER THR CYS VAL ILE LEU MET PHE'
    lipidselect = 'resname {}'.format(lpd_res)## is this smart way to select the 
    #carbselect = ''
    #ionselect = "resname SOD CLA Na Cl" ## Add more later
    watselect = 'resname HOH TP3 TIP3 WAT TIP'
    
    print('Parsing the PDB to find molecular components')
    print('Found {} chains in the structure'.format(PDB.n_chains))
    
    moltype_dict = {}
    sys_chain = []
    lipidpres=False
    for chain in PDB.topology.chains:
        print(chain.index)
        chain_subset = PDB.topology.subset(PDB.topology.select('chainid {}'.format(chain.index)))
        prot = chain_subset.select(protselect)
        rna = chain_subset.select(rnaselect)
        dna = chain_subset.select(dnaselect)
        lip = chain_subset.select(lipidselect)
        
        wat = chain_subset.select(watselect)
        #ion = chain_subset.select(ionselect)
        if prot.size>0:
            name = 'PROT'
            print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, chain_subset.n_atoms))
            selatom = protselect
            sys_chain.append(str(chain.index))
            
        elif rna.size>0:
            name = 'RNA'
            print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, chain_subset.n_atoms))
            selatom = rnaselect
            sys_chain.append(str(chain.index))
            
        elif dna.size>0:
            name = 'DNA'
            print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, chain_subset.n_atoms))
            selatom = dnaselect
            sys_chain.append(str(chain.index))
        
        elif lip.size>0:
            name = 'LIPID'
            print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, chain_subset.n_atoms))
            selatom = lipidselect
            lipidpres= True
            sys_chain.append(str(chain.index))
        #elif ion.size>0:
        #    name = 'ION'
        #    selatom = ionselect
        #    print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, dna.size))
            
        elif wat.size>0:
            name='WAT'
            selatom = watselect
            print('chainID {} is a {} molecule with {} atoms'.format(chain.index, name, chain_subset.n_atoms))
        else:
            del name
            del selatom
            continue
        
        moltype_dict['{}-{}'.format(name,chain.index)]={'SELSTR': 'chainid {}'.format(chain.index),
                                                        'SELATM': PDB.topology.select('chainid {}'.format(chain.index))}
    if (len(sys_chain)>1)&(not lipidpres):
        moltype_dict['SOLUTE'] = {'SELSTR': 'chainid {}'.format(' '.join(sys_chain)),
                                  'SELATM': PDB.topology.select('chainid {}'.format(' '.join(sys_chain)))}
    return moltype_dict 

def _build_ldtypes(ld_flag_value):
    """
    
    """
    global lipid_acylmap
    lipid_acylmap = {'CH':{'acyl1':'cholesterol','acyl2':None,'n_carbon':[]},
                     'DL':{'acyl1':'lauric','acyl2':'lauric','n_carbon':[12,12]},
                     'DM':{'acyl1':'mystric','acyl2':'mystric','n_carbon':[14,14]},
                     'DP':{'acyl1':'palmitic','acyl2':'palmitic','n_carbon':[16,16]},
                     'PY':{'acyl1':'palmitic','acyl2':'palmitoleic','n_carbon':[16,16]},
                     'PS':{'acyl1':'palmitic','acyl2':'stearic','n_carbon':[16,18]},
                     'PO':{'acyl1':'palmitic','acyl2':'oleic','n_carbon':[16,18]},
                     'PL':{'acyl1':'palmitic','acyl2':'linoleic','n_carbon':[16,18]},
                     'DY':{'acyl1':'palmitoleic','acyl2':'palmitoleic','n_carbon':[16,16]},
                     'YO':{'acyl1':'palmitoleic','acyl2':'oleic','n_carbon':[16,18]},
                     'OY':{'acyl1':'oleic','acyl2':'palmitoleic','n_carbon':[18,16]},
                     'DS':{'acyl1':'stearic','acyl2':'stearic','n_carbon':[18,18]},
                     'SO':{'acyl1':'stearic','acyl2':'oleic','n_carbon':[18,18]},
                     'SL':{'acyl1':'stearic','acyl2':'linoleic','n_carbon':[18,18]},
                     'DO':{'acyl1':'oleic','acyl2':'oleic','n_carbon':[18,18]},
                     }
    global lipid_HGmap
    lipid_HGmap = {'L1':{'name':'cholesterol'},
                   'PG':{'HEx':['OC3','OC2'], 'NEx':['C3','C2','C1','C11','C12','C13'],'name':'glycerol'},
                   'PI':{'name':'inositol','HEx':['O2','O3','OP42','O6'], 'NEx':['C3','C2','C1','C11','C12','C13','C14','C15','C16']},  ## Currently only supporting PIP2(4,5)
                   'PA':{'name':'atidic','HEx':[], 'NEx':['C3','C2','C1']},
                   'PC':{'name':'choline','HEx':[], 'NEx':['C3','C2','C1','C11','C12','C13','C14','C15']},
                   'PS':{'name':'serine','HEx':['N'], 'NEx':['C3','C2','C1','C11','C12']},
                   'PE':{'name':'ethanolamine','HEx':['N'], 'NEx':['C3','C2','C1','C11','C12']}
                  }
    
    #lipid_HGtype = {'C':'cardiolipin', 'P':'phosphate'}
    
    ld_type_fdeut = np.array([ld_type.split('-') for ld_type in ld_flag_value.split(',')])
    ld_type_df = pd.DataFrame(ld_type_fdeut)
    ld_type_df = pd.concat([ld_type_df,ld_type_df[1].str.split('/', expand=True).rename(columns={0:'acyl1',1:'acyl2',2:'head'})],axis=1)
    ld_type_df = ld_type_df.rename(columns={0:'resName',1:'deuteration string'})
    ld_type_df['acyl_key'] = (ld_type_df['resName'].astype(str).str.get(0)+ld_type_df['resName'].astype(str).str.get(1))
    ld_type_df['HG_key'] = (ld_type_df['resName'].astype(str).str.get(2)+ld_type_df['resName'].astype(str).str.get(3))
    ld_type_df['acyl1_type'] = pd.Series(data=[lipid_acylmap[lact]['acyl1'] for lact in ld_type_df['acyl_key'].values])
    ld_type_df['acyl2_type'] = pd.Series(data=[lipid_acylmap[lact]['acyl2'] for lact in ld_type_df['acyl_key'].values])
    ld_type_df['HG_type'] = pd.Series(data=[lipid_HGmap[lhg]['name'] for lhg in ld_type_df['HG_key'].values])
    ld_type_df = ld_type_df.set_index('resName')
    ## Make the dictionary that maps the atom names to the lipid domains
    res2LipidDomain = {ldrname:{} for ldrname in ld_type_df.index.values}
    for indx, ldt_val in ld_type_df.iterrows():
        if 'sterol' in ldt_val.acyl1_type:
            print(ldt_val.acyl1_type)
            continue
        acyl1_nc, acyl2_nc = lipid_acylmap[ldt_val.acyl_key]['n_carbon']
        acyl1_name2chain = {'C3{}'.format(ac1):'acyl1' for ac1 in range(2,acyl1_nc+1)}
        acyl2_name2chain = {'C2{}'.format(ac1):'acyl2' for ac1 in range(2,acyl2_nc+1)}
        hg_name2chain = {atmn:'head' for atmn in lipid_HGmap[ldt_val.HG_key]['NEx']}
        res2LipidDomain[indx] = {**acyl1_name2chain, **acyl2_name2chain, **hg_name2chain}
        
    return ld_type_df, res2LipidDomain

def _hydrogen_select_solute(topdf, atmselect, moltypesel='PROT'):
    ## integrated into the Structure class (Future)
## inherited by each solute type 
## solvent type handled differently 
    moltopdf = topdf.loc[atmselect]
    ## Problem w/ Serial is that it can duplicate, use index
    ExHDatomdf = moltopdf[(moltopdf['element'] =='O')|(moltopdf['element'] =='N')|(moltopdf['element'] =='S')|(moltopdf['element'] =='P')].index
    NExHDatomdf = moltopdf[(moltopdf['element'] =='C')].index
    HYDatomdf = moltopdf[(moltopdf['element'] =='H')].index
    
    ## Exchangeable hydrogens
    hexd_index = np.array(list(itertools.product(ExHDatomdf,HYDatomdf)))
    hexdiff_index = np.diff(hexd_index,axis=1)
    hexd_index = hexd_index[np.where((hexdiff_index<=3)&(hexdiff_index>0))[0]]
    hexdist_index = np.where(md.compute_distances(PDB, hexd_index)<0.125)[1]
    ## Residue Check
    hex_reslist = [[PDB.topology.atom(i).residue, PDB.topology.atom(j).residue] for i,j in hexd_index[hexdist_index]]
    
    if len(hex_reslist) > 0:
        if np.all(np.array(hex_reslist)[:,0] == np.array(hex_reslist)[:,1]):
            print("Exchangeable hydrogen covalent bonding information found: All N/O/S/P-H bonds selected are part of the same residue")
            hex_hyd = [[PDB.topology.atom(i), i, PDB.topology.atom(j), j] for i,j in hexd_index[hexdist_index]]
        else:
            print("Some exchangeable hydrogens are not part of the same residue:")
            print("Check topology/PDB")
            #raise SystemExit()
            print(hex_reslist,np.all(np.array(hex_reslist)[:,0] == np.array(hex_reslist)[:,1]))
            print(np.array(hex_reslist)[np.where(np.array(hex_reslist)[:,0] != np.array(hex_reslist)[:,1])[0]])
            return None
    else:
        print("No exchangeable hydrogens found in {}".format(moltypesel))
        hex_hyd = []
    ## Non-Exchangeable Hydrogens
    
    nexd_index = np.array(list(itertools.product(NExHDatomdf, HYDatomdf)))
    nexdiff_index = np.diff(nexd_index,axis=1) ## indexing check
    print('selecting based off serial indexing')
    nexd_index = nexd_index[np.where((nexdiff_index<=3)&(nexdiff_index>0))[0]]
    print('computing distances')
    nexdist_index = np.where(md.compute_distances(PDB, nexd_index)<0.125)[1] ## distance check
    ## Residue Check
    print('residue check')
    nex_reslist = [[PDB.topology.atom(i).residue, PDB.topology.atom(j).residue] for i,j in nexd_index[nexdist_index]]
    if np.all(np.array(nex_reslist)[:,0] == np.array(nex_reslist)[:,1]):
        print("Non-exchangeable covalent hydrogen bonding information found: All C-H bonds selected are part of the same residue")
        nex_hyd = [[PDB.topology.atom(i), i, PDB.topology.atom(j), j] for i,j in nexd_index[nexdist_index]]
    else:
        print("Some non-exchangeable hydrogens are not in the same residue")
        print("Check topology/PDB")
        return None
    
    exhdf = pd.DataFrame(hex_hyd,columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    exhdf['N_HydBonds'] = 0
    exhdf['N_Contacts'] = 0
    nexhdf = pd.DataFrame(nex_hyd,columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    nexhdf['H2DConvert'] = False 
    #print(nexhdf.head())
    if mainargs.segmental_deuteration:
         
         nexhdf_HI = nexhdf.set_index('H_Index')
         sd_index = np.where(np.array(segNEX_Occ)[atmselect,1]==1.0)[0]
         pdbsd_atomindex = np.array(segNEX_Occ)[atmselect,0][sd_index]
         #print(nexhdf.loc[sd_index[:10]])
         #print(nexhdf_HI.loc[sd_index[:10]])
         #print(nexhdf_HI.loc[7])
         nexhdf_HI.loc[pdbsd_atomindex,'H2DConvert'] = True
         nexhdf = nexhdf_HI.reset_index()[nexhdf.columns]
 
    if np.any(exhdf['H_Index'].duplicated()):
        print('Found duplicated hydrogen indices in the exchangeable hydrogen selection')
        print('Check topology/structure for defects')
        print(exhdf[exhdf['H_Index'].duplicated()])
        return None
    if np.any(nexhdf['H_Index'].duplicated()):
        print('Found duplicated hydrogen indices in the non-exchangeable hydrogen selection')
        print('Check topology/structure for defects')
        print(nexhdf['H_Index'].duplicated())
        return None
        
    return exhdf, nexhdf

def _add_LipidDomains():
    
    NExHDF_t['HA_Donor_resNum'] = NExHDF_t['HA_Donor'].astype('str').str.split('-', expand=True)[0].str.slice(4).astype(int)
    NExHDF_t['HA_Donor_resName'] = NExHDF_t['HA_Donor'].astype('str').str.split('-', expand=True)[0].str.slice(start=0,stop=4)
    NExHDF_t['HA_Donor_atmname'] = NExHDF_t['HA_Donor'].astype('str').str.split('-', expand=True)[1]
    
    if ExHDF_SF.shape[0]>0:
        ExHDF_SF['HA_Donor_resNum'] = ExHDF_SF['HA_Donor'].astype('str').str.split('-', expand=True)[0].str.slice(4).astype(int)
        ExHDF_SF['HA_Donor_resName'] = ExHDF_SF['HA_Donor'].astype('str').str.split('-', expand=True)[0].str.slice(start=0,stop=4)
        ExHDF_SF['HA_Donor_atmname'] = ExHDF_SF['HA_Donor'].astype('str').str.split('-', expand=True)[1]
    
    ## use the res2LipidDomain dictionary to map the atom names to the non-exchangeable hydrogen definitions
    NExHDF_t['HA_Donor_Domain'] = 'head' ## always set to head: its easier to define acyl chains than head groups. everything that is missed is a head group?
    for ldresn in LD_type_df.index.values:
        nexdf_resdf = NExHDF_t[NExHDF_t['HA_Donor_resName'] == ldresn].copy()
        if ('sterol' in LD_type_df.loc[ldresn,'acyl1_type'])&(LD_type_df.loc[ldresn,'acyl2']==None):
            continue
        NExHDF_t.loc[nexdf_resdf.index,'HA_Donor_Domain'] = nexdf_resdf['HA_Donor_atmname'].apply(lambda atmn:RES2LipidDomain[ldresn][atmn])
    
    
def _add_structurefactors(exhdf, selstr, solute=False):
    
    try:
        hbond_array, nhb_counts = calc_hbonds(selstr, solbool=solute)
        if solute:
            ## only need to do this due to multimers of the same system for solute
            exhdf['BondPair'] = (exhdf['HA_Donor'].astype('str') 
                                 + ' - ' + exhdf['ExH'].astype('str')
                                 + ' - ' + exhdf['H_Index'].astype('str')
                                )
        else:
            exhdf['BondPair'] = (exhdf['HA_Donor'].astype('str') 
                                 + ' - ' + exhdf['ExH'].astype('str')
                                 # + ' - ' + exhdf['H_Index'].astype('str')
                                )
        
        exhdf = exhdf.set_index('BondPair')
        exhdf.loc[nhb_counts.index, 'N_HydBonds'] = nhb_counts.values
        exhdf = exhdf.reset_index()
        
    except KeyError as err:
        print('Key error returned during Hydrogen Bond calculation: -- {}'.format(err))
        print('Most likely reason is no hydrogen bonds found in the particular chain')
        print('Setting number of hydrogen bonds to zero ')
        exhdf['N_HydBonds'] = 0 
    except ValueError as err:
        print('Value Error occured during hydrogen bond calculation:--{}'.format(err))
        print(len(nhb_counts.index), len(nhb_counts), exhdf.loc[nhb_counts.index].shape[0])
    
    ncontacts_solute = calc_ncontacts(selstr,
                                      exhdf['HA_Donor_Index'].values)
    
    
    exhdf['N_Contacts'] = ncontacts_solute.flatten().astype(int)
    exhdf['PFactor'] = exhdf.apply(lambda rr: calc_pfactor_BVModel(rr.N_HydBonds, rr.N_Contacts), axis=1)
    return exhdf

def isPROT(molstr)->bool:
    return ('PROT' in molstr)
def isDNA(molstr)->bool:
    return ('DNA' in molstr)
def isRNA(molstr)->bool:
    return ('RNA' in molstr)

def _add_solute_h2dindex(phxf=0.95,nahxf=1.0,lhxf=1.0):
    
    for key in MolHYDDF_Dict.keys():
        print(key)
        temp_dict = MolHYDDF_Dict[key] 
        temp_dict['ExH']['H2DConvert'] = False ## Always reset to false
        
        
        ## Update pfactor with whole complex calculations only if SOLUTE is defined
        
        if 'SOLUTE' in list(MolHYDDF_Dict.keys()):
            chpf = (temp_dict['ExH'])[['H_Index','PFactor']].set_index('H_Index')
            solpf = (MolHYDDF_Dict['SOLUTE']['ExH'])[['H_Index','PFactor']].set_index('H_Index').loc[chpf.index]
            #print(len(solpf),len(chpf))
            temp_dict['ExH']['PFactor'] = ((chpf + solpf)/2).values
        
        if ('PROT' in key):  
            ## Selecting the random indices for the exchange
            pgD2O=moldict_parse[key]['gD2O'][NpgD2O]
            hxex_h2d_index, nex_h2d_index = _random_select_protein(pgD2O, psD2O, 
                                                                    temp_dict['ExH'], 
                                                                    temp_dict['NExH'], 
                                                                    nrandom_sets,
                                                                    hexfactor=phxf,
                                                                    segdeut=mainargs.segmental_deuteration)
            
        elif ('RNA' in key) | ('DNA' in key):
            ## select indices for the exchange for nucleic acids
            hxex_h2d_index, nex_h2d_index = _random_select_nucleic(0.0, psD2O, 
                                                                    temp_dict['ExH'], 
                                                                    temp_dict['NExH'], 
                                                                    nrandom_sets,
                                                                    hexfactor=nahxf,
                                                                    segdeut=mainargs.segmental_deuteration)
            
        elif ('LIPID' in key):
            hxex_h2d_index, nex_h2d_index = _random_select_lipid(LD_type_df, psD2O,
                                                                 temp_dict['ExH'],
                                                                 temp_dict['NExH'],
                                                                 nrandom_sets,
                                                                 hexfactor=lhxf,
                                                                 segdeut=mainargs.segmental_deuteration)
        else:
            continue
                
        temp_dict['hexh2d'] = hxex_h2d_index
        temp_dict['nexh2d'] = nex_h2d_index
        MolHYDDF_Dict[key] = temp_dict
    
    return None

def ConvertH2D():
    
    ex_topdf = topologydf.copy()
    ex_topdf = ex_topdf.merge(coorddf, how='left',left_index=True,right_index=True)
    ex_bfdf = bfactordf.copy() ## Holds BFactors for writing them 
    ## Convert the names for the different molecule types
    
    for key in MolHYDDF_Dict.keys():
        
        if (isPROT(key))|(isRNA(key))|(isDNA(key)): 
            ### Exchangeable Hydrogens
            hex_h2ddf = MolHYDDF_Dict[key]['ExH'].copy()
            hdx_index = MolHYDDF_Dict[key]['hexh2d'][pdbnum]
            hex_h2ddf['PFactor'] = 0.0
            ex_bfdf['PFactor']=0.0
            if (not mainargs.noHDX):
                hex_h2ddf = MolHYDDF_Dict[key]['ExH'].copy()
                hdx_index = MolHYDDF_Dict[key]['hexh2d'][pdbnum]
                hex_h2ddf.loc[hdx_index, 'H2DConvert']=1
                ex_topdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values, ['name','element']] = 'D'
                #print(ex_bfdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values].shape, len(hex_h2ddf['PFactor'].values))
                ex_bfdf.loc[hex_h2ddf['H_Index'].values, 'PFactor'] = hex_h2ddf['PFactor'].values
                ex_bfdf.loc[hex_h2ddf['H_Index'].values, 'H2DConvert'] = hex_h2ddf['H2DConvert'].values
                #ex_topdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values, 'element'] = 'D'
            
            ### NonExchangeable Hydrogens
            nex_h2df = MolHYDDF_Dict[key]['NExH'].copy()
            nex_index = MolHYDDF_Dict[key]['nexh2d'][pdbnum]
            nex_h2df.loc[nex_index,'H2DConvert']=2
            ex_topdf.loc[nex_h2df.loc[nex_index, 'H_Index'].values, ['name','element']] = 'D'
            ex_bfdf.loc[nex_h2df['H_Index'].values, 'H2DConvert'] = 2
            #ex_topdf.loc[nex_h2df.loc[nex_index, 'H_Index'].values, 'element'] = 'D'
        elif ('LIPID' in key):
            
            if MolHYDDF_Dict[key]['ExH'].size > 0:
                hex_h2ddf = MolHYDDF_Dict[key]['ExH'].copy()
                hex_h2ddf['PFactor'] = 0.0
                hdx_index = MolHYDDF_Dict[key]['hexh2d'][pdbnum]
                hex_h2ddf.loc[hdx_index, 'H2DConvert']=1
                ex_topdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values, ['name','element']] = 'D'
                #print(ex_bfdf.loc[hex_h2ddf.loc[hdx_index, 'H_Index'].values].shape, len(hex_h2ddf['PFactor'].values))
                ex_bfdf.loc[hex_h2ddf['H_Index'].values, 'PFactor'] = hex_h2ddf['PFactor'].values
                ex_bfdf.loc[hex_h2ddf['H_Index'].values, 'H2DConvert'] = hex_h2ddf['H2DConvert'].values
            
            nex_h2df = MolHYDDF_Dict[key]['NExH'].copy()
            nex_index = MolHYDDF_Dict[key]['nexh2d'][pdbnum]
            nex_h2df.loc[nex_index,'H2DConvert']=2
            ex_topdf.loc[nex_h2df.loc[nex_index, 'H_Index'].values, ['name','element']] = 'D'
            ex_bfdf.loc[nex_h2df['H_Index'].values, 'H2DConvert'] = 2
    ### Solvent Hydrogens:
    solvex_h2df = SolventHDF.copy()
    solvex_h2df.loc[solv_index, 'H2DConvert'] = 3
            
    ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values,['name','element']] = 'D'
    ex_bfdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'H2DConvert'] = 3
    #ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'element'] = 'D'
    ex_bfdf['PFactor'] =  ex_bfdf['PFactor'].fillna(0.0)  
    ex_bfdf['H2DConvert'] = ex_bfdf['H2DConvert'].fillna(0.0).astype(int)
    ex_topdf = ex_topdf.merge(ex_bfdf, how='left',left_index=True,right_index=True)
    
    return ex_topdf

def ConvertH2D_SolvOnly():
    
    ex_topdf = topologydf.copy()
    ex_topdf = ex_topdf.merge(coorddf, how='left',left_index=True,right_index=True)
    ex_bfdf = bfactordf.copy() ## Holds BFactors for writing them 

    ### Solvent Hydrogens:
    solvex_h2df = SolventHDF.copy()
    solvex_h2df.loc[solv_index, 'H2DConvert'] = 3
            
    ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values,['name','element']] = 'D'
    ex_bfdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'H2DConvert'] = 3
    #ex_topdf.loc[solvex_h2df.loc[solv_index, 'H_Index'].values, 'element'] = 'D'
    ex_bfdf['PFactor'] =  0
    ex_bfdf['H2DConvert'] = ex_bfdf['H2DConvert'].fillna(0.0).astype(int)
    ex_topdf = ex_topdf.merge(ex_bfdf, how='left',left_index=True,right_index=True)
    
    return ex_topdf

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
                        type=str, default="0.0,100.0",
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
    parser.add_argument("--ldtype_fracd", type=str,
                        help="fraction of lipid acyl chain C-H hydrogens to exchange to deuterium")
    
    parser.add_argument("--solvent_only", default=False, action='store_true',
                        help="run the code to deuterate the solvent only")
    
    parser.add_argument("--randsolv", default=False, action='store_true',
                        help="random solvent hydrogen to deuterium exchange model")

    parser.add_argument("--solute_randhex", default=False, action='store_true',
                       help="randomize the hd exchange in the solute i.e. equal weights versus p-factor weigthed")
 
    parser.add_argument("--fraction_hdx", default=0.95, action='store_true',
                        help="total fraction of exchangeable hydrogens to be exchanged")
   
    parser.add_argument("--noHDX", default=False, action='store_true', 
                        help="Turn off hydrogen deuterium exchange in the solute")

    parser.add_argument("pdbfile", nargs='*',
                        help="solvated pdb file with the system of interest")
    
    return parser

### GLOBAL Variable descriptions
USAGE="{} [options] [inputpdb]".format(sys.argv[0])
DESCRIPTION="Deuterate a system of interest for contrast matching prediction"
VERSION="{} version 1.0"
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
    
    SOD_indx = topologydf[topologydf['name']=='Na+'].index
    topologydf.loc[SOD_indx,'name'] = 'SOD'
    topologydf.loc[SOD_indx,'resName']='SOD'
    CLA_indx = topologydf[topologydf['name']=='Cl-'].index
    topologydf.loc[CLA_indx,'name'] = 'CLA'
    topologydf.loc[CLA_indx,'resName']='CLA'
    print(mainargs.solvent_only, mainargs.noHDX)    
    
    if (mainargs.ldtype_fracd):
        print('expecting {} lipid types in the topology'.format(len(mainargs.ldtype_fracd.split(','))))
        LD_type_df, RES2LipidDomain = _build_ldtypes(mainargs.ldtype_fracd)
        ld_type_seqsel = ' '.join(LD_type_df.index.values)
    else:
        ld_type_seqsel = 'POPC'
    
    ## How to target the different chains for growth conditions?
    ## Parse by / in the growth D2O item
    ## ADD "0.0" to make all the same growth
    ## Should use the chain  ordering in PDB
    pgrowthD2O_list = [pg.split(',') for pg in mainargs.percent_growthD2O.split('/')]
    print(pgrowthD2O_list)
    pg_lenm = np.array([len(pgs) for pgs in pgrowthD2O_list])
    Nunique_pgd2o = np.unique(pg_lenm).shape[0]
    if Nunique_pgd2o > 1:
        max_npg = max(pg_lenm)
        nzerogrowth = abs(pg_lenm - max_npg)
        pg_samelength = [np.concatenate([pgrowthD2O_list[ipg],['0.0']*pgl]) for ipg, pgl in enumerate(nzerogrowth)]
    else:
        pg_samelength = pgrowthD2O_list
    pgD2O_multi = np.array(pg_samelength).astype(float)
    #pgrowthD2O_list = [float(pg) for pg in mainargs.percent_growthD2O.split(',')]
    ##
    ## Solvent conditions always stay the same? Yes!
    psolventD2O_list = [float(pg) for pg in mainargs.percent_solventD2O.split(',')]
    print(pgD2O_multi)
    print(psolventD2O_list)
    
    moldict_parse = _parse_topology(lpd_res=ld_type_seqsel)
    if mainargs.segmental_deuteration:
         ## segmental PDB test
         segdeut_pdb = mainargs.pdbfile[1]
         segNEX_Occ = _read_segmentaldeut_pdb(segdeut_pdb)
         #print(segNEX_Occ)
    #print(moldict_parse)
    ## Loop generates all the exchangeable hydrogens, non-exchangeable hydrogen selection for 
    ## 
    if (not mainargs.solvent_only) & (not mainargs.noHDX):
        MolHYDDF_Dict = {}
        for key in moldict_parse.keys():
            #print(key)
            if ('PROT' in key)|('RNA' in key)|('DNA' in key):
                ExHDF_t, NExHDF_t = _hydrogen_select_solute(topologydf, moldict_parse[key]['SELATM'])
                ## Struture Calculation
                ExHDF_SF = _add_structurefactors(ExHDF_t, moldict_parse[key]['SELSTR'])
                
            elif 'LIPID' in key:
                ExHDF_SF, NExHDF_t = _hydrogen_select_solute(topologydf,  moldict_parse[key]['SELATM'])
                _add_LipidDomains()
                    ## Full structure p-factor calculation    
            if key == 'SOLUTE':
                ExHDF_t, NExHDF_t = _hydrogen_select_solute(topologydf, moldict_parse[key]['SELATM'])
            ## Struture Calculation
                ExHDF_SF = _add_structurefactors(ExHDF_t, moldict_parse[key]['SELSTR'],True)
            
            MolHYDDF_Dict[key] = {'ExH':ExHDF_SF, 'NExH':NExHDF_t}
        bool_prot = [('PROT' in ky) for ky in list(moldict_parse.keys())]
        #print(bool_prot)
        prot_keys = np.array(list(moldict_parse.keys()))[bool_prot]
        #print(prot_keys)
        for pkey, pgd2o_list in zip(prot_keys, pgD2O_multi):
            pdict = moldict_parse[pkey]
            pdict['gD2O'] = pgd2o_list
            moldict_parse[pkey] = pdict
            print(moldict_parse[pkey])
    elif (mainargs.solvent_only) & mainargs.noHDX:
        print("This is a solvent only run without Hydrogen Deuterium Exchange in the solute")
        MolHYDDF_Dict = {}
        for key in moldict_parse.keys():
            #print(key)
            if ('PROT' in key)|('RNA' in key)|('DNA' in key):
                ExHDF_t, NExHDF_t = _hydrogen_select_solute(topologydf, moldict_parse[key]['SELATM'])
                ## Struture Calculation
                ExHDF_SF = _add_structurefactors(ExHDF_t, moldict_parse[key]['SELSTR'])
            elif 'LIPID' in key:
                ExHDF_SF, NExHDF_t = _hydrogen_select_solute(topologydf,  moldict_parse[key]['SELATM'])
                _add_LipidDomains()
            if key == 'SOLUTE':
                ExHDF_t, NExHDF_t = _hydrogen_select_solute(topologydf, moldict_parse[key]['SELATM'])
                ## Struture Calculation
                ExHDF_SF = _add_structurefactors(ExHDF_t, moldict_parse[key]['SELSTR'])
            
            MolHYDDF_Dict[key] = {'ExH':ExHDF_SF, 'NExH':NExHDF_t}
        bool_prot = [('PROT' in ky) for ky in list(moldict_parse.keys())]
        #print(bool_prot)
        prot_keys = np.array(list(moldict_parse.keys()))[bool_prot]
        #print(prot_keys)
        for pkey, pgd2o_list in zip(prot_keys, pgD2O_multi):
            pdict = moldict_parse[pkey]
            pdict['gD2O'] = pgd2o_list
            moldict_parse[pkey] = pdict
            print(moldict_parse[pkey])
        #print(ExHDF_t.head(),flush=True)
    else:
        print("This is a solvent only deuteration run.")
        ## Add the gD2O lists to Mol
        
    ## Solvent Selection and Manipulation
    solvent_H = PDB.topology.select('(resname HOH or resname TP3 or resname WAT) and type H')
    solvent_O = PDB.topology.select('(resname HOH or resname TP3 or resname WAT) and type O')
    SolventHDF = pd.DataFrame(index=np.arange(len(solvent_H)),columns=['HA_Donor', 'HA_Donor_Index','ExH','H_Index'])
    SolventHDF['H2DConvert'] = False 
    SolventHDF['H_Index'] = solvent_H
    SolventHDF['ExH'] = SolventHDF['H_Index'].apply(lambda solvh: PDB.topology.atom(solvh))
    SolventHDF['HA_Donor_Index'] = np.vstack([solvent_O.T,solvent_O.T]).T.flatten()
    SolventHDF['HA_Donor'] = SolventHDF['HA_Donor_Index'].apply(lambda solvo: PDB.topology.atom(solvo))
    
    pdbprefixloc = '/'.join((mainargs.pdbfile[0].split('/')[:-1]))
    if mainargs.prefix:
        pbdsuffix=mainargs.prefix
    else:
        pbdsuffix = mainargs.pdbfile[0].split('/')[-1].split('.pdb')[0]
    print('Setting the pdb prefix to {}'.format(pbdsuffix))
    temp_pdbdir = '{}/Template_PDBs/'.format(pdbprefixloc)
    try:
        os.mkdir(temp_pdbdir)
    except:
        print('directory already exists')
   # print(mainargs.nrand_pdb)
    ## Begin write loop
    for psD2O in psolventD2O_list:
        err_check = False
        ## loop over deuteration schemes, not chains
        for NpgD2O, prot_gd2ol in enumerate(pgD2O_multi.T):
            print(prot_gd2ol)
            if mainargs.nrand_pdb:
                nrandom_sets=mainargs.nrand_pdb
            else:
                nrandom_sets = random_sets(psD2O, pgD2O_multi.max())
        
        ## Update the dictionary after each loop to reselect the random indices for the solute elements
        ## at a given growth and solvent condition
            if (not mainargs.solvent_only):
                
                try:
                    _add_solute_h2dindex(phxf=mainargs.fraction_hdx)
                except Exception as e:
                    print(e)
                    print('This may be a structural defect in the PDB')
                    raise SystemExit()
            elif (mainargs.solvent_only) & (mainargs.noHDX):

                try:
                    _add_solute_h2dindex(phxf=mainargs.fraction_hdx)
                except Exception as e:
                    print(e)
                    print('This may be a structural defect in the PDB')
                    raise SystemExit()        
            
            ## Solvent random selection
            if mainargs.randsolv:
                print("using the random solvent model")
                solvex_h2d_index = _random_select_solvent_RANDH2D(psD2O, SolventHDF, nrandom_sets)
            else:
                solvex_h2d_index = _random_select_solvent(psD2O, SolventHDF, nrandom_sets)
            pgD2O_string = '-'.join([str(int(pg)) for pg in prot_gd2ol])
            for pdbnum, solv_index in enumerate(solvex_h2d_index[:nrandom_sets]):
            ## Every iteration copy the dataframes:
                if (not mainargs.solvent_only):
                    ex_topdf = ConvertH2D()
                elif (mainargs.solvent_only) & (mainargs.noHDX):
                    ex_topdf = ConvertH2D()
                else:
                    ex_topdf = ConvertH2D_SolvOnly()

                if mainargs.segmental_deuteration:
                    Dpdb_fname = '{}/{}_{}-sD2O_SegDeut_NEx{:04}.pdb'.format(temp_pdbdir,pbdsuffix, int(psD2O), pdbnum+1)
                else:    
                    Dpdb_fname = '{}/{}_{}-sD2O_{}-gD2O_NEx{:04}.pdb'.format(temp_pdbdir, pbdsuffix, int(psD2O), pgD2O_string, pdbnum+1)            
                write_pdb(Dpdb_fname, ex_topdf)
            
        
    
                
