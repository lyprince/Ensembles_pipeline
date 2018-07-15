'''
Author : Luke Prince
Date : 14th July 2018

Non rigid motion correction of miniscope video

Adapted from demo_pipeline_CNMFE by CaImAn team
'''

from os import system, path
import yaml
import argparse
from caiman.motion_correction import motion_correct_oneP_rigid, motion_correct_oneP_nonrigid

def get_args():
    parser = argparse.ArgumentParser(description = 'Non-rigid motion correction of miniscope videos')
    parser.add_argument('animal', help='animal ID', nargs=1)
    parser.add_argument('session', help='session: test, trainA, or trainB', nargs=1)
    parser.add_argument('--base_dir', dest = 'base_dir', help='Base directory to find files', default='/home/luke/Documents/Projects/RichardsPostdoc/Ensembles/CA1_imaging/')
    parser.add_argument('-n', '--num_processes', help'Number of processes to add to cluster', type=int, default=8)
    parser.add_argument('-b', '--backend', help='Cluster backend', default='local')
    
    
if __name__ == '__main__':
    
    get_args()
    
    args = get_args()
    
    cell_info = yaml.load(open('./cell_metadata.yaml'))
    
    animal    = args.animal[0]
    session   = args.session[0]
    timestamp = cell_info[animal][session]
    fileext   = cell_info[animal]['orig_file_ext']
    frate     = cell_info[animal]['frame_rate']
    
    filename  = args.base_dir + '%s/%s_%s_%s%s'%(animal, timestamp, animal, session, fileext)
    
    fnames = args.input
    parameters = yaml.load(open('./parameters.yaml'))['motion_correction']
    
    gSig_filt  = parameters['gSig_filt']    # size of gaussian filter. Change this if algorithm doesn't work
    max_shifts = parameters['max_shifts']   # maximum allowed rigid shift
    splits_rig = parameters['splits_rig']   # for parallelisation split movies in num_splits chunks across time
    strides    = parameters['strides']      # start a new patch for pw-rigid motion correction every x pixels
    overlaps   = parameters['overlaps']     # overlap between patches. Patch size = strides + overlaps
    splits_els = parameters['splits_els']   # for parallelisation split movies in num_splits chunks across time. 
                                            # Tip: make sure len(movie)/num_splits > 100

    upsample_factor_grid = parameters['upsample_factor_grid']  # upsample factor to avoid smearing when merging patches
    max_deviation_rigid  = parameters['max_deviation_rigid']   # maximum deviation allowed for patch w.r.t. rigid shifts.

    c, dview, n_processes = cm.cluster.setup_cluster(backend=args.backend,             # use this one
                                                     n_processes=args.num_processes,   # number of processes to use, if you go out of memory try to reduce this one
                                                     )
    
    # Motion correction rigid to obtain template
    mc = motion_correct_oneP_rigid(fnames, gSig_filt = gSig_filt, max_shifts = max_shifts, splits_rig = splits_rig,
                                   dview = dview, save_movie = False)
    new_templ = mc.total_template_rig
    
    bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
    
    # Motion correction non-rigid

    mc = motion_correct_oneP_nonrigid(fnames, gSig_filt = gSig_filt,
                                      max_shifts=max_shifts, strides=strides, overlaps=overlaps, 
                                      splits_els=splits_els, upsample_factor_grid=upsample_factor_grid,
                                      max_deviation_rigid=max_deviation_rigid, splits_rig=None, 
                                      save_movie=do_motion_correction_nonrigid,
                                      new_templ=new_templ, border_nan=bord_px)
    
    bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                             np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    
    fname_new = cm.save_memmap([mc.fname_tot_els], base_name=dirname+'memmap_', order = 'C', border_to_0=bord_px, dview=dview)