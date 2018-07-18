'''
Author : Luke Prince
Date : 16th July 2018

Concatenate a set of movies to visualise effects of processing steps
'''

from os import system, path
import yaml
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from caiman import load_memmap

import pdb

def get_args():
    parser = argparse.ArgumentParser(description = 'Concatenate a set of movies at various stages of processing')
    parser.add_argument('input', help='input videos', nargs='+')
    parser.add_argument('-s', '--speed', help='output speed up factor', type=int, default=8)
    parser.add_argument('-b', '--border', help='Pad border around separate videos by a defined number of pixels', type = int, default=20)
    return parser.parse_args()
    
    
if __name__ == '__main__':
    
    def check_dims(paths):
        frame_count = []
        frame_width = []
        frame_height = []
        
        for p in paths:
            rawpath, ext = path.splitext(p)
            basename = path.basename(p)
            
            if ext == '.avi':
                cap = cv2.VideoCapture(p)
                frame_count.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                frame_width.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                frame_height.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                cap.release()
            
            elif ext == '.mmap':
                basename_desc = basename[basename.find('memmap'):]
                _0, _1, height, _2, width, _3, _4, _5, _6, _7, count, _ext = basename_desc.split('_')
                frame_count.append(int(count))
                frame_width.append(int(width))
                frame_height.append(int(height))
                
        assert len(set(frame_count))==1, 'Videos do not have the same number of frames'
        assert len(set(frame_width))==1, 'Videos do not have the same frame width'
        assert len(set(frame_height))==1, 'Videos do not have the same frame height'
        
        return frame_count[0], frame_width[0], frame_height[0]
    
    def check_animal_session(paths):
        animals = []
        sessions = []

        for p in paths:
            dirname = path.dirname(path.realpath(p))
            timestamp, animal, session = path.basename(p).split('_')[:3]
            session = path.splitext(session)[0]
            animals.append(animal)
            sessions.append(session)

        assert len(set(animals))==1, 'Videos not from same animal'
        assert len(set(sessions))==1, 'Videos not from same session'
        
        return dirname, timestamp, animals[0], sessions[0]
    
    def frameGenerator(path_to_video):
        rawpath, ext = path.splitext(path_to_video)
        
        if ext == '.avi':
            cap = cv2.VideoCapture(path_to_video)
            gen = (cap.read()[1][:,:,0] for ix in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            
        elif ext == '.mmap':
            mmap, dims, T = load_memmap(path_to_video)
            mmap = mmap.T.reshape((T,) + dims, order='F')
            gen = (frame for frame in mmap)
        
        return gen
    
    args = get_args()
    
    dirname, timestamp, animal, session = check_animal_session(args.input)
    frame_count, frame_width, frame_height = check_dims(args.input)
    frameGenerators = [frameGenerator(path_to_video) for path_to_video in args.input]
    
    total_frame_height = frame_height + 2*args.border
    total_frame_count = frame_count
    total_frame_width = frame_width*len(args.input) + (len(args.input)+1)*args.border
    
    frame_base = np.zeros((total_frame_height, total_frame_width))
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    
    output_filename = dirname + '/%s_%s_%s_comparison.avi'%(timestamp, animal, session)
    out = cv2.VideoWriter(output_filename, fourcc, 40, (total_frame_width, total_frame_height), isColor=False)
    
    movieTypes = ['Original', 'Motion Corrected', 'Source Extracted']

    
    for frames in tqdm(zip(*frameGenerators)):
        total_frame = frame_base.copy()
        bottomLeftTexts = []
        
        for ix,frame in enumerate(frames):
            xs = args.border*(ix+1) + frame_width*ix
            xe = args.border*(ix+1) + frame_width*(ix+1)
            ys = args.border
            ye = args.border + frame_height
            
            total_frame[ys:ye,xs:xe] = frame
            total_frame = total_frame.astype('uint8')
            cv2.putText(total_frame, movieTypes[ix], org = (xs + 10, ys + 10), 
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=255, lineType=2)
                  
        out.write(total_frame)

    out.release()
        