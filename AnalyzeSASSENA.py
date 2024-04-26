#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:24:56 2022

@author: 9cq
"""

import itertools
import csv
import numpy as np
import pandas as pd
import mdtraj as md
import glob

import h5py
import itertools
import scipy.interpolate as intp
from scipy.optimize import fsolve,curve_fit

import argparse
import sys
import os

def _convert_GD20_NEXH_re(pgrowth_d2o):
    """
    
    Empirical Function that will be determined by HDX-MS data:
    Converts the growth conditions of the system to determine the deuteration level of non-exchangeable hydrogens
    Quadratic u
    array([0.40595313, 0.46273678])
    """
    C1= 0.40595313
    C2= 0.46273678
    #C1 = 0.42
    #C2 = 0.44
    return C1 * (pgrowth_d2o/100) + C2 * np.power(pgrowth_d2o/100,2) 

def solv_matchpoint(i0_data,sld=False):
    #print(i0_data)
    if sld:
        spacing = np.linspace(-0.060,6.7,101)
    else:
        spacing = np.linspace(0,100,101)
    
    print(spacing)
    print(i0_data.index, i0_data.values.flatten())
    fsansm = intp.splrep(i0_data.index, i0_data.values.flatten())
    ## What happens if I already know the inflection point?
    inflection = intp.splev(spacing, fsansm).min()
    inflection_index = spacing[np.where(intp.splev(spacing, fsansm) == inflection)[0]]
    print("The inflection sqrt(I(0)) and % solvent D2O point are:")
    print(inflection, inflection_index)
    abs_distance = abs(inflection - i0_data.values.flatten()) ## absolute distance between the inflection point value and the i0 results
    closest_points = np.argsort(abs_distance)[:2] ## sort distances lowest to highest: take the first two indices
    
    intperp_derivates = pd.DataFrame(index=spacing,
                                 data=np.array(intp.spalde(spacing, fsansm))[:,0:4])
    der_clpoints = intperp_derivates.loc[i0_data.index[closest_points],1]
    print("Derivative at the closest points are:")
    print(der_clpoints)
    
    if abs_distance[closest_points[0]]>5e-5:
        ## if the absolute distance is far away from the tolerance of 5e-5 then you should have two points
        ## to easily transition over the data
        #intperp_derivates = pd.DataFrame(index=spacing,
        #                         data=np.array(intp.spalde(spacing, fsansm))[:,0:4])
        #der_clpoints = intperp_derivates.loc[i0_data.index[closest_points],1]
        inv_point = der_clpoints[der_clpoints>0.0].index
        if len(inv_point) == 0:
            inv_point = [der_clpoints.index[0]]
        print("The inversion point is : {}".format(inv_point))
        index_invert = np.where(i0_data.index==inv_point[0])[0]
        i0_data.iloc[index_invert[0]:] *= -1
        
    else:
        ## How to decide if I should take the + 1 position or the 0 position if close to the inflection point?
        print("Closest point is very close to the true inflection point already. Deciding which point to invert. ")
        print("The derivative at the closest point, {}, is: {}".format(i0_data.index[closest_points[0]],
                                                                       intperp_derivates.loc[i0_data.index[closest_points[0]],1]))
        ## if the slope at the closest point is negative? 
        ### Take the next point in the sD2O series
        ## Else:
        ## Keep the closest point:
        dClosePoints = intperp_derivates.loc[i0_data.index[closest_points[0]],1]
        if dClosePoints < 0:
            print("The derivative at the closest point is negative. Taking the next point to invert")
            i0_data.iloc[(closest_points[0]+1):] *= -1
        else:
            print("The derivate at the closest point is positive. Using this point as the inversion")
            i0_data.iloc[(closest_points[0]):] *= -1
    
    lin_fit = lambda x,m,b: m*x+b
    i0_fit_params, i0_fit_errors = curve_fit(lin_fit, xdata=i0_data.index[:-1], ydata=i0_data.values.flatten()[:-1])
    print(i0_fit_params)
    plin_fit = lambda x:lin_fit(x, *i0_fit_params)
    #print(spacing[np.where(intp.splev(spacing, fsansm) == inflection)[0]])
    matchpoint = fsolve(plin_fit, spacing[np.where(intp.splev(spacing, fsansm) == inflection)[0]])
    
    return matchpoint, i0_data

def _match_series(i0_datadf, growth_series, solvent_series):
    match_series=[]
    i0_inverse_dataDF = pd.DataFrame(index=np.array(solvent_series), columns=np.array(growth_series))
    #print(i0_inverse_dataDF)
    for gd2o in growth_series:
        i0_solv_g=i0_datadf[i0_datadf['gD2O']==gd2o][['sD2O','sqrt(I(0))']].set_index('sD2O')
        #print(gd2o)
        gd2o_point, i0_data_inv = solv_matchpoint(i0_solv_g)
        #print(i0_data_inv)
        i0_inverse_dataDF.loc[i0_data_inv.index.values, gd2o]=i0_data_inv.values.flatten()
        match_series.append([gd2o, gd2o_point[0]])
    return match_series, i0_inverse_dataDF

def _match_series_SLD(i0_datadf, growth_series, solvent_series):
    match_series=[]
    i0_inverse_dataDF = pd.DataFrame(index=np.array(solvent_series), columns=np.array(growth_series))
    #print(i0_inverse_dataDF)
    
    for gd2o in growth_series:
        
        i0_solv_g=i0_datadf[i0_datadf['gD2O']==gd2o][['SLD','sqrt(I(0))']].set_index('SLD')
        #print(gd2o)
        gd2o_point, i0_data_inv = solv_matchpoint(i0_solv_g, sld=True)
        #print(i0_data_inv)
        i0_inverse_dataDF.loc[i0_data_inv.index.values, gd2o]=i0_data_inv.values.flatten()
        match_series.append([gd2o, gd2o_point[0]])
        
    return match_series, i0_inverse_dataDF

def _read_sassena(filesdf):
    
    with h5py.File(filesdf.iloc[0,0],'r') as sass_out:
    
        sass_keys = list(sass_out.keys())
        print(sass_keys)
        #qvecndx = np.where(sass_keys=='qvectors')
        #print(qvecndx)
        qvectors = np.array((sass_out.get('qvectors')))
        axis_qvec = ~np.all(qvectors==0,axis=0)
        #print(qvectors, axis_qvec,flush=True)
        qvectors = qvectors[:,np.where(axis_qvec)[0][0]]
        
    sasmindex = filesdf.set_index(['sD2O','gD2O','nex'])
    #print(sasmindex, qvectors)
    sasdf = pd.DataFrame(index=sasmindex.index, columns=np.sort(qvectors)).fillna(0.0).T
                         
    for index, h5f in filesdf.iterrows():
        #print(h5f)
        with h5py.File(h5f.filename,'r') as sassh5:
            sass_keys = list(sassh5.keys())
            qvectors = np.array((sassh5.get('qvectors')))
            axis_qvec = ~np.all(qvectors==0,axis=0)
            sqv = np.argsort(qvectors[:,np.where(axis_qvec)[0][0]])
            fq = np.array((sassh5.get('fq')))[sqv]
            sasdf[(h5f.sD2O, h5f.gD2O, h5f.nex)] = fq[fq!=0.0]
    
    sassenadf_ave = sasdf.groupby(level=[0,1], axis=1, sort=False).mean()
    sassenadf_std = sasdf.groupby(level=[0,1], axis=1, sort=False).std()
    
    return sassenadf_ave, sassenadf_std

def _transform_SASSENA(sassenadf):
    
    sassenadf_mapit = (sassenadf.iloc[0].apply(np.sqrt).reset_index().rename(columns={0.0:'sqrt(I(0))'}))
    sassenadf_mapit['sD2O'] = sassenadf_mapit['index'].str.split('_',expand=True)[0].str.split('-',expand=True)[1].astype(float)
    sassenadf_mapit['gD2O'] = sassenadf_mapit['index'].str.split('_',expand=True)[1].str.split('-',expand=True)[1].astype(float)
    
    return sassenadf_mapit 


def init_argparse():
    
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
    
    parser.add_argument("-s","--percent_solventD2O",
                        type=str, default="0.0,100.0",
                        help="percent D2O present in the experimental solvent media"
                        )
    
    parser.add_argument('-m',"--match_solve", type=str, default="sD2O", 
                        help = """This flag determines what to actually solve for. The options are sD2O or gD2O.
                         sD2O (default) : solve for the solvent matchpoint given a particular gD2O condition.
                         gD2O : solve for the growth percent to match out at a particular sD2O condition.""")
                         
    parser.add_argument("-p","--prefix",
                        type=str, 
                        help="""file prefix naming. Will use glob to grab the file list with the prefix""")
    
    parser.add_argument("-u","--suffix", type=str, 
                        help="""file naming suffix. Will use glob to grab with the suffix""")
    
    parser.add_argument("-r","--regex", type=str,
                        help="""python regex string to capture the file names for analysis""")
                                
    parser.add_argument("-n","--nrand_pdb",
                        type=int,
                        help="Number of SASSENA h5 files calculated from the random template PDBs"
                        )
    
    parser.add_argument( "-l","--flist", nargs='*',
                        help="list of SASSENA output files to read in. Not recommended for large file lists.")
    
    return parser

USAGE="{} [options] [templatepdb] [mdfile]".format(sys.argv[0])
DESCRIPTION="Deuterate a system of interest for contrast matching prediction"
VERSION="{} version 0.0.4"

if __name__=='__main__':
    
    cmlparse = init_argparse()
    args = cmlparse.parse_args()
    print(args)
   
    if (args.suffix):
        filelist = glob.glob('*{}.h5'.format(args.suffix))
                            
    elif (args.prefix):
        filelist = glob.glob('{}*.h5'.format(args.prefix))
    
    elif (args.regex):
        filelist = glob.glob(args.regex)
    
    elif (args.flist):
        filelist = args.flist
    else:
        raise SystemExit("Use -p, -s, or -l to define the file lists")
        
    sd2o_list = [];
    gd2o_list = []; 
    nex_index = [];
    for fname in filelist:
        fname_split = fname.split('_')
        for fspl in fname_split:
            if 'sD2O' in fspl:
                sd2o_list.append(int(fspl.split('-')[0]))
            elif 'gD2O' in fspl:
                gd2o_list.append(int(fspl.split('-')[0]))
            elif 'NEx' in fspl:
                nex_index.append(int(fspl.split('NEx')[1].lstrip('0').split('.')[0]))
            else:
                continue;
        
    #print(np.vstack([filelist, sd2o_list, gd2o_list , nex_index]).T.shape)
    filedf = pd.DataFrame(np.vstack([filelist, sd2o_list, gd2o_list , nex_index]).T,
                          columns=['filename','sD2O','gD2O','nex'])
    filedf['sD2O'] = filedf['sD2O'].astype(int)
    filedf['gD2O'] = filedf['gD2O'].astype(int)
    filedf['nex'] = filedf['nex'].astype(int)
    filedf = filedf.sort_values(['sD2O','nex','gD2O']).reset_index(drop=True)
    
    
    sasave, sasstd = _read_sassena(filedf)
    sasave.to_csv('SASSENA_Average.csv')
    sasstd.to_csv('SASSENA_StandardDeviation.csv')
    sasave_map = sasave.iloc[0].apply(np.sqrt)
    sasave_map = sasave_map.reset_index().rename(columns={0:'sqrt(I(0))'})
    matchout, sasave_i0_inv = _match_series(sasave_map, filedf['gD2O'].unique(), filedf['sD2O'].unique())
    sasave_i0_inv = sasave_i0_inv.reset_index().rename(columns={'index':'sD2O',0:'sqrt(I(0))'})
    sasave_i0_inv.to_csv('sqrtI0_Inverted.csv')
    print(matchout)
    print('The percent solvent D2O matchout condition for {} growth D2O is {}'.format(matchout[0][0],matchout[0][1]))
    
         
    
        
                
                
                
       
      
