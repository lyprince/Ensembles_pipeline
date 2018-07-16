'''
Author : Luke Prince
Date : 16th July 2018
'''

from os import system, path
import yaml
import argparse

import cv2
import numpy as np
from caiman import load_memmap

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
                timestamp, animal, session, _0, _1, height, _2, width, _3, _4, _5, _6, _7, count, _ext = basename.split('_')
                frame_count.append(int(count))
                frame_width.append(int(width))
                frame_height.append(int(height))
        
        print(frame_count)
                
        assert len(set(frame_count))==1, 'Videos do not have the same number of frames'
        assert len(set(frame_width))==1, 'Videos do not have the same frame width'
        assert len(set(frame_height))==1, 'Videos do not have the same frame height'
        
        return frame_count[0], frame_width[0], frame_height[0]
    
    
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
    
    frame_count, frame_width, frame_height = check_dims(args.input)
    frameGenerators = [frameGenerator(path_to_video) for path_to_video in args.input]