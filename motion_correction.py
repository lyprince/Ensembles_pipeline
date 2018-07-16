'''
Author : Luke Prince
Date : 14th July 2018

Non rigid motion correction of miniscope video

Adapted from demo_pipeline_CNMFE by CaImAn team
'''

from os import system, path
import yaml
import argparse
import numpy as np
from caiman import save_memmap
from caiman.cluster import setup_cluster
from caiman.motion_correction import motion_correct_oneP_rigid, motion_correct_oneP_nonrigid

def get_args():
    parser = argparse.ArgumentParser(description = 'Non-rigid motion correction of miniscope videos')
    parser.add_argument('animal', help='animal ID', nargs=1)
    parser.add_argument('session', help='session: test, trainA, or trainB', nargs=1)
    parser.add_argument('--base_dir', dest = 'base_dir', help='Base directory to find files', default='/home/luke/Documents/Projects/RichardsPostdoc/Ensembles/CA1_imaging/')
    parser.add_argument('-n', '--num_processes', help = 'Number of processes to add to cluster', type=int, default=8)
    parser.add_argument('-b', '--backend', help='Cluster backend', default='local')
    parser.add_argument('-r', '--redo', help= 'Redo motion correction', action='store_true')
    parser.set_defaults(redo=False)
    return parser.parse_args()
    
if __name__ == '__main__':
    
    args = get_args()
    cell_info = yaml.load(open('./cell_metadata.yaml'))
    
    animal    = args.animal[0]
    session   = args.session[0]
    timestamp = cell_info[animal][session]['timestamp']
    fileext   = '.avi'
    frate     = cell_info[animal]['frame_rate']
    completed = cell_info[animal][session]['motion_correction']['completed']
    filename  = args.base_dir + '%s/%s_%s_%s%s'%(animal, timestamp, animal, session, fileext)
    
    if not completed or args.redo:
    
        fnames    = [filename]

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

        c, dview, n_processes = setup_cluster(backend=args.backend,             # use this one
                                                         n_processes=args.num_processes,   # number of processes to use, if you go out of memory try to reduce this one
                                                         )
        
        if path.exists(path.splitext(filename)[0]+'_mc_template.npy'):
            
            mc_template = np.load(path.splitext(filename)[0]+'_mc_template.npy')
            bord_px = cell_info[animal][session]['motion_correction']['bord_moco']
        else:
            # Motion correction rigid to obtain template
            mc = motion_correct_oneP_rigid(fnames, gSig_filt = gSig_filt, max_shifts = max_shifts, splits_rig = splits_rig,
                                           dview = dview, save_movie = False)
            mc_template = mc.total_template_rig
            np.save(path.splitext(filename)[0]+'_mc_template.npy',mc_template)
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
            cell_info[animal][session]['motion_correction']['bord_moco'] = int(bord_px)

        plt.figure()
        plt.imshow(mc_template)       #% plot template
        plt.title('Filtered template')
        fig.savefig(path.splitext(filename)[0]+'_mc_template.svg')
        
        # Motion correction non-rigid

        mc = motion_correct_oneP_nonrigid(fnames, gSig_filt = gSig_filt,
                                          max_shifts=max_shifts, strides=strides, overlaps=overlaps, 
                                          splits_els=splits_els, upsample_factor_grid=upsample_factor_grid,
                                          max_deviation_rigid=max_deviation_rigid, splits_rig=None, 
                                          save_movie=False, dview = dview,
                                          new_templ=mc_template, border_nan=bord_px)

        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                 np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        
        cell_info[animal][session]['motion_correction']['bord_cnmf'] = int(bord_px)
        
        fname_new = save_memmap([mc.fname_tot_els], base_name=path.splitext(filename)[0]+'_memmap',
                                   order = 'C', border_to_0=bord_px, dview=dview)
        
        system('rm %s'%mc.fname_tot_els[0])
        
        cell_info[animal][session]['motion_correction']['completed'] = True
        
        yaml.dump(cell_info, open('./cell_metadata.yaml', 'w'))
        
        dview.terminate()
        
    else:
        frame_count = cell_info[animal][session]['frame_count']
        frame_width = cell_info[animal][session]['frame_width']
        frame_height = cell_info[animal][session]['frame_height']
        
        filename_new = path.splitext(filename)[0] + '_memmap_d1_%i_d2_%i_d3_1_order_C_frames_%i_.mmap'%(frame_height, frame_width, frame_count)
        print(filename_new)
        assert path.exists(filename_new), 'Path does not exist, try to redo preprocessing with -r option'
        print('Motion correction step for %s_%s already completed'%(animal, session))