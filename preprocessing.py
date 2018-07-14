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
    parser.add_argument('--fps', help='frames per second', default=20)
    return parser.parse_args()

if __name__ == '__main__':
    
    def get_video_dims(cap):
        '''
        Retrieve video dimensions from cv2.VideoCapture object
        
        Input:
            - cap: cv2.VideoCapture object
        
        Output:
            - frame_count: int, number of frames
            - frame_width: int, number of pixels in frame width
            - frame_height: int, number of pixels in frame height
        '''
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))       
        return (frame_count, frame_width, frame_height)

    def get_vid_crop_lims(filename, crop_thresh=40):
        '''
        Find x,y limits where the mean fluorescence is always above a defined threshold value

        Input:
            - filename: string, path to video
            - crop_thresh: int, fluorescence threshold to find x,y limits to crop to
        Output:
            - xlims: tuple of 2 ints, x-axis pixels to crop to
            - ylims: tuple of 2 ints, y-axis pixels to crop to
        '''
        cap = cv2.VideoCapture(filename)
        frame_count, frame_width, frame_height = get_video_dims(cap)
        xs = np.inf
        xe = 0
        ys = np.inf
        ye = 0

        y = np.arange(frame_height)
        x = np.arange(frame_width)
        
        for ix in tqdm(range(frame_count), desc = 'Obtaining crop limits'):
            ret, frame = cap.read()     
            if ret == True:
                frame_xy = frame[:,:,0]
                
                xf = frame_xy.mean(axis=0)
                yf = frame_xy.mean(axis=1)
                
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
                
            else:
                break
                
        cap.release()

        return (xs, xe), (ys, ye), frame_count
    
            
    def process_and_convert(filename, ds_factor, xlims, ylims, fps):
        
        '''
        Downsample, crop, and convert video to avi
        
        Input:
            filename  : string, path to input video
            ds_factor : int, downsampling factor
            xlims     : tuple of 2 ints, x-axis limits to crop to
            ylims     : tuple of 2 ints, y-axis limits to crop to
            fps       : int or float, frame rate of output video
        '''
        
        cap = cv2.VideoCapture(filename)
        frame_rate = fps
        frame_count, frame_width, frame_height = get_video_dims(cap)
        if xlims is not None and ylims is not None:
            xs,xe = xlims
            ys,ye = ylims
        else:
            xs, ys = 0, 0
            xe, ye = frame_width, frame_height
        
        filename_new = path.splitext(filename)[0] + '.avi'
        out = cv2.VideoWriter(filename=filename_new, fourcc=0, fps = fps/ds_factor, frameSize=(xe-xs, ye-ys), isColor=True)
        frame_set = []
        
        for ix in tqdm(range(frame_count), 'Downsampling and cropping, and saving to avi'):
            ret, frame = cap.read()            
            frame_set.append(frame[ys:ye, xs:xe]) # Cropping
            if len(frame_set) == ds_factor or ix == (frame_count-1):
                frame_set = np.array(frame_set)
                frame_proc = np.round(frame_set.mean(axis=0)).astype('uint8') # Downsampling
                out.write(frame_proc)

                frame_set = []
                    
        
        cap.release()
        out.release()
                              
    
    args = get_args()
    
    cell_info = yaml.load(open('./cell_metadata.yaml'))
    
    animal    = args.animal[0]
    session   = args.session[0]
    timestamp = cell_info[animal][session]
    fileext   = cell_info[animal]['orig_file_ext']
    frate     = cell_info[animal]['frame_rate']
    
    filename  = args.base_dir + '%s/%s_%s_%s%s'%(animal, timestamp, animal, session, fileext)
    assert path.exists(filename), 'Path does not exist'
    print('Downsampling, cropping, and converting %s: %s to .avi'%(animal, session))
    
    if args.crop:
        xlims, ylims, frame_count = get_vid_crop_lims(filename, crop_thresh = args.crop_thresh)
    else:
        xlims = None
        ylims = None
    process_and_convert(filename=filename, ds_factor=args.downsample, xlims=xlims, ylims=ylims, fps=args.fps)