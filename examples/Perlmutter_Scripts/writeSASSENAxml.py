#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:38:16 2022

@author: 9cq
"""

from lxml.etree import ElementTree as ET
import lxml.etree as xmet

import argparse
import sys
import os

USAGE="{} [options] [templatepdb] [mdfile]".format(sys.argv[0])
DESCRIPTION="Deuterate a system of interest for contrast matching prediction"
VERSION="{} version 0.0.4"

def init_argparse() ->argparse.ArgumentParser:

    parser = argparse.ArgumentParser(usage = USAGE, 
                                      description=DESCRIPTION
                                      )
    parser.add_argument("-v","--version",action='version',
                        version=VERSION.format(parser.prog)
                        )
    
    ##trajectory options
    parser.add_argument("-f","--first", type=int, default=0,
                        help="snapshot index of the first snapshot you want to average from" )
    parser.add_argument("-l","--last",  type=int, default=1,
                        help="snapshot index of the last snapshot you want to average to")
    parser.add_argument("-s","--stride", type=int, default=1,
                        help="steps between snapshots of the simulation")
    
    
    ##sample selections
    parser.add_argument("--sel_name", action="append", nargs="+",
                        help="selection names: should be given as a list i.e. --sel_name water solute ")
    parser.add_argument("--sel_index", action="append", nargs="+",
                        help="""selection indexes: pdb based indexes for the selections.
                        Should be ordered the same as --sel_name.
                        The indices should be colon delimited for from index:to index i.e. --sel_index 1:5 6:10""")
                        
    ## Scattering Options
    
    parser.add_argument("-nq",type=int, default=101, help="number of q-values to calculate")
    parser.add_argument("--min_q", type=float,default=0.0, help="Minimum q value")
    parser.add_argument("--max_q", type=float, default=0.6, help="Maximum q value")
    parser.add_argument("-res", type=int, default=1000, help="Resolution: # of q-vectors per magnitude of q")
    parser.add_argument("-b","--background", type=float, default=0.0, help="Contrast Background Scattering Length Density")
    
    parser.add_argument("-k","--kappas", action="append", nargs="+",
                        help="kappa values for the background correction. Each selection from --sel_name should have a kappa value.")
    
    parser.add_argument("-o","--out",type=str, default="signal.h5", help="name of the output .h5 file for signal.")
    parser.add_argument("-t","--stage_target", type=str, help="selection target for the scattering. ")
    
    parser.add_argument("-x","--xml_out", type=str, default="sassena.xml", help="name of the xml output")
    requiredNamed = parser.add_argument_group('required named arguments')
    

    requiredNamed.add_argument("-i", "--template_pdb", type=str, nargs=1,  required=True, 
                        help="solvated template pdb for explicit deuteration scheme")
    
    requiredNamed.add_argument("-d", "--dcd", type=str, nargs=1, required=True, 
                        help="trajectory file formatted as a dcd. Must have at least one frame")
    
    
    return parser

if __name__=='__main__':
    
    cmlparse = init_argparse()
    args = cmlparse.parse_args()
    #print(args)
    
    ## checks for the correct variables and ordering 
    if len(args.sel_name[0])!=len(args.sel_index[0]):
        print("Selection names and selection indices don't have the same length")
        raise SystemExit("Please provide the same number of selection groups and indexes. See -h for details.")
    
    
    if (args.kappas == None) & (len(args.sel_name[0])>0):
        print("""kappa values not provided for the selection names. Default values are 1.0.
               Make sure this is what you want.""")
    
        args.kappas = [['1.0']*len(args.sel_name[0])]
        
    if (args.kappas != None) & (len(args.kappas[0]) > len(args.sel_name[0])):
        print("More kappa values provided than selection names.")
        raise SystemExit("Please provide equal to or less than the number of selections for kappas.")
        
    elif (args.kappas != None) & (len(args.kappas[0]) < len(args.sel_name[0])):
        print("less kappa values than selections. Adding the rest with default values of 1.0")
        kps = args.kappas[0]
        #print(kps,['1.0']*(len(args.sel_name[0])-len(args.kappas[0])),type(kps[0]))
        
        for kp in ['1.0']*(len(args.sel_name[0])-len(args.kappas[0])):
            kps.append(kp)
        #print(kps)
        args.kappas = [kps]
    
    sassenaxml = xmet.Element('root')
    sassenaTree = ET(sassenaxml)
    ## Primary children
    xml_database = xmet.SubElement(sassenaxml, 'database')
    xml_sample = xmet.SubElement(sassenaxml, 'sample')
    xml_scattering = xmet.SubElement(sassenaxml, 'scattering')
    

    ## Database
    xml_typedatabase = xmet.SubElement(xml_database,'type')
    xml_typedatabase.text = 'file'
    xml_filedatabase = xmet.SubElement(xml_database,'file')
    ## make a variable? # 
    xml_filedatabase.text = "/pscratch/sd/a/ach15/SCOMAPXD/database/db-neutron-coherent.xml"

    ## Sample 
    ## Sample Structure
    sample_structure = xmet.SubElement(xml_sample,'structure')
    sample_struct_file = xmet.SubElement(sample_structure,'file')
    sample_struct_file.text = args.template_pdb[0]
    sample_struct_format = xmet.SubElement(sample_structure,'format')
    sample_struct_format.text = "pdb"

    ## Sample Trajectory 
    sample_frameset = xmet.SubElement(xml_sample,'framesets')
    sample_fset =  xmet.SubElement(sample_frameset,'frameset')
    sample_fset_file = xmet.SubElement(sample_fset,'file')
    sample_fset_file.text = args.dcd[0]
    sample_fset_format = xmet.SubElement(sample_fset,'format')
    sample_fset_format.text = "dcd"
    sample_fset_first = xmet.SubElement(sample_fset,'first')
    sample_fset_first.text = str(args.first)
    sample_fset_last = xmet.SubElement(sample_fset,'last')
    sample_fset_last.text = str(args.last)
    sample_fset_stride = xmet.SubElement(sample_fset,'stride')
    sample_fset_stride.text = str(args.stride)

    ## Sample Selections
    ## If Not provided Default is System and weird things happen with Kappa values in scattering
    sample_selections = xmet.SubElement(xml_sample,'selections')
    
    for sname, ndx_vals in zip(args.sel_name[0],args.sel_index[0]):
        sample_sel = xmet.SubElement(sample_selections,'selection')
        sample_sel_type = xmet.SubElement(sample_sel,'type')
        sample_sel_type.text = 'range'
        sample_sel_from =  xmet.SubElement(sample_sel,'from')
        sample_sel_from.text = "{}".format(ndx_vals.split(':')[0])
        sample_sel_to =  xmet.SubElement(sample_sel,'to')
        sample_sel_to.text = "{}".format(ndx_vals.split(':')[1])
        sample_sel_name = xmet.SubElement(sample_sel,'name')
        sample_sel_name.text = sname

    ## Scattering DSP type :: square for coherent solution scattering
    scatter_type = xmet.SubElement(xml_scattering,'type') ## all or self: All for coherent
    scatter_type.text = "all" 
    scatter_dsp = xmet.SubElement(xml_scattering,'dsp')
    scatter_dsp_type = xmet.SubElement(scatter_dsp,'type')
    scatter_dsp_type.text = "square"
    ## Not Necessary for 
    scatter_dsp_method = xmet.SubElement(scatter_dsp,'method')
    scatter_dsp_method.text = "fftw"

    ## Scattering Vectors
    scatter_vectors = xmet.SubElement(xml_scattering, 'vectors')
    scatter_vectors_type = xmet.SubElement(scatter_vectors, 'type')
    scatter_vectors_type.text = "scans"
    scatter_vec_scans = xmet.SubElement(scatter_vectors, 'scans')
    scatter_vscan = xmet.SubElement(scatter_vec_scans, 'scan')
    scatter_vscan_from = xmet.SubElement(scatter_vscan, 'from')
    scatter_vscan_from.text = "{:.3f}".format(args.min_q)
    scatter_vscan_to = xmet.SubElement(scatter_vscan, 'to')
    scatter_vscan_to.text = "{:.3f}".format(args.max_q)
    scatter_vscan_points = xmet.SubElement(scatter_vscan, 'points')
    scatter_vscan_points.text = "{}".format(args.nq)
    scatter_vscan_base =xmet.SubElement(scatter_vscan, 'base')
    scatter_vscan_bqx = xmet.SubElement(scatter_vscan_base, 'x')
    scatter_vscan_bqx.text = "0" 
    scatter_vscan_bqy = xmet.SubElement(scatter_vscan_base, 'y')
    scatter_vscan_bqy.text = "0" 
    scatter_vscan_bqz = xmet.SubElement(scatter_vscan_base, 'z')
    scatter_vscan_bqz.text = "1"

    ## Scattering Average
    scatter_ave = xmet.SubElement(xml_scattering, 'average')
    scatter_ave_orient = xmet.SubElement(scatter_ave, 'orientation')
    scatter_aveOr_type = xmet.SubElement(scatter_ave_orient, 'type')
    scatter_aveOr_type.text = "vectors"
    scatter_aveOr_Vec = xmet.SubElement(scatter_ave_orient, 'vectors')
    scatter_avevec_type = xmet.SubElement(scatter_aveOr_Vec, 'type')
    scatter_avevec_type.text = "sphere"
    scatter_Or_alg = xmet.SubElement(scatter_aveOr_Vec, 'algorithm')
    scatter_Or_alg.text = "boost_uniform_on_sphere"
    scatter_Or_res = xmet.SubElement(scatter_aveOr_Vec, 'resolution')
    scatter_Or_res.text = str(args.res)

    ## Scattering Background
    scatter_back = xmet.SubElement(xml_scattering, 'background')
    scatter_backfact = xmet.SubElement(scatter_back, 'factor')
    scatter_backfact.text = "{:.6f}".format(args.background)
    scatter_kappas = xmet.SubElement(scatter_back, 'kappas')
    for sname, ndx_vals in zip(args.sel_name[0], args.kappas[0]):
        print("setting the Kappa value for Selection:{} to {}".format(sname,ndx_vals))
        sample_kap = xmet.SubElement(scatter_kappas,'kappa')
        sample_kap_sel = xmet.SubElement(sample_kap,'selection')
        sample_kap_sel.text = sname
        sample_kap_val =  xmet.SubElement(sample_kap,'value')
        sample_kap_val.text = ndx_vals
    
    
    ## Scattering Signal....
    scatter_signal = xmet.SubElement(xml_scattering, 'signal')
    scatter_signal_file = xmet.SubElement(scatter_signal, 'file')
    if args.out[-3:] != ".h5": ## Potentially problematic if more than one . in file name
        print("file extension for the signal results is not h5.")
        raise SystemExit("Exiting change name to have extension h5. See -h or --help for details")
        
    scatter_signal_file.text = args.out
    scatter_signal_fqt = xmet.SubElement(scatter_signal, 'fqt')
    scatter_signal_fqt.text = "false"
    scatter_signal_fq0 = xmet.SubElement(scatter_signal, 'fq0')
    scatter_signal_fq0.text = "true"
    scatter_signal_fq = xmet.SubElement(scatter_signal, 'fq')
    scatter_signal_fq.text = "true"
    scatter_signal_fq2 = xmet.SubElement(scatter_signal, 'fq2')
    scatter_signal_fq2.text = "false"

    ## Staging Target Selection:: ## write two out: one to get the background factor and another to calculate
    if args.stage_target:
        xml_stager = xmet.SubElement(sassenaxml, 'stager')
        scatter_target = xmet.SubElement(xml_stager, 'target')
        scatter_target.text = str(args.stage_target)
    
    with open(args.xml_out, mode="w") as outfile:
        outfile.write(xmet.tostring(sassenaxml, pretty_print=True).decode())
