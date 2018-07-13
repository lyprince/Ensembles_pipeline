'''
Author : Luke Prince
Date : 13th July 2018

Preprocessing of Josselyn-Frankland miniscope videos: downsampling, cropping, and converting to .avi

Adapted from minipipe and pre-cnmfe in chendoscope-minipipe (Lina Tran and Andrew Mocle, 2017)
'''

from os import system, path
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import cv2

def get_args():
    parser = argparse.ArgumentParser(description = 'Downsample, crop, and convert videos to .avi')
    parser.add_argument('animal', help='animal ID', nargs=1)
    parser.add_argument('session', help='session: test, trainA, or trainB', nargs=1)
    parser.add_argument('--base_dir', dest = 'base_dir', help='Base directory to find files', default='/home/luke/Documents/Projects/RichardsPostdoc/Ensembles/CA1_imaging/')
    parser.add_argument('-c', '--crop', dest='crop', help='Crop videos', action='store_true')
    parser.set_defaults(crop=False)
    parser.add_argument('-ct', '--crop_thresh', dest='crop_thresh', type=int, default=40)
    parser.add_argument('-d', '--downsample', help='downsample factor, default is 4', type=int, default=4)
    parser.add_argument('--cores', help='cores to use, default is 1', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    cell_info = yaml.load(open('./cell_metadata.yaml'))
    animal    = str(args.animal[0])
    session   = str(args.session[0])
    timestamp = cell_info[animal][session]
    fileext   = cell_info[animal]['orig_file_ext']
    frate     = cell_info[animal]['frame_rate']
    
    filename  = args.base_dir + '%s/%s_%s_%s%s'%(animal, timestamp, animal, session, fileext)
    print(filename)
    assert path.exists(filename), 'Path does not exist'

    
def downsample(vid, ds_factor, xlims=None, ylims=None):
    '''
    Downsample video by ds_factor.
    
    If xlims and ylims are not None, crop video to these limits also

    Input:
        - vid: numpy array, video
        - ds_factor: int, downsample factor
        - xlims (optional): tuple of ints, x-index of crop limits
        - ylims (optional): tuple of ints: y-index of crop limits
        
    Output:
        - vid_ds: numpy array, downsampled video
    '''
    dims = vid[0].shape
    
    if xlims is not None:
        xs, xe = xlims
    else:
        xs = 0
        xe = dims[1] - 1
    
    if ylims is not None:
        ys, ye = ylims
    else:
        ys = 0
        ye = dims[0] - 1
        
    
    dims = vid[0].shape
    vid_ds = np.zeros((int(len(vid)/ds_factor), ye-ys, xe-xs))

    frame_ds = 0
    for frame in tqdm(range(0, len(vid), ds_factor), desc='Downsampling'):
        if frame + ds_factor <= len(vid):
            stack = np.array(vid[frame:frame+ds_factor])[:,ys:ye,xs:xe,0]
            vid_ds[frame_ds, :, :] = np.round(np.mean(stack, axis=0))
            frame_ds += 1

        else:
            continue

    return vid_ds

def get_crop_lims(vid, crop_thresh=40):
    '''
    Find x,y limits where the mean fluorescence is always above a defined threshold value
    
    Input:
        - vid: numpy array, video
        - crop_thresh: int, fluorescence threshold to find x,y limits to crop to
    Output:
        - xlims: tuple of 2 ints, x-axis pixels to crop to
        - ylims: tuple of 2 ints, y-axis pixels to crop to
    '''
    dims = vid[0].shape
    xs = np.inf
    xe = 0
    ys = np.inf
    ye = 0

    y = np.arange(dims[0])
    x = np.arange(dims[1])
    
    for frame in vid:
        frame = np.array(frame)[:,:,0]

        xf = frame.mean(axis=0)
        yf = frame.mean(axis=1)

        x_thresh = x[xf>=crop_thresh]
        y_thresh = y[yf>=crop_thresh]

        if x_thresh[0] < xs:
            xs = x_thresh[0]

        if x_thresh[-1] > xe:
            xe = x_thresh[-1]

        if y_thresh[0] < ys:
            ys = y_thresh[0]

        if y_thresh[-1] > ye:
            ye = y_thresh[-1]
            
        return (xs, xe), (ys, ye)
    
def save_to_avi(vid, fps, filename):
    
    total_frames, height, width = vid.shape
    container = av.open(filename, 'w')
    stream = container.add_stream('rawvideo', rate=fps)
    stream.height = height
    stream.width = width
    stream.pix_fmt = 'bgr24'
    
    for frame in vid:
        # Convert frame to RGB uint8 values
        frame = frame.astype('uint8')
        frame = np.repeat(np.reshape(frame, newshape=(frame.shape[0], frame.shape[1], 1)), repeats=3, axis=2)
        
        # Encode frame into stream

        frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        for packet in stream.encode(frame):
            container.mux(packet)
    
    # Flush Stream
    for packet in stream.encode():
        container.mux(packet)

    # Close file
    container.close()