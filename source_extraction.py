'''
Author : Luke Prince
Date : 16th July

Neuronal source extraction from motion-corrected video using CNMFE.

Adapted from demo_pipeline_CNMFE by CaImAn team.
'''

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Neuronal source extracted from motion corrected video using CNMFE')
    parser.add_argument('animal', help='animal ID', nargs=1)
    parser.add_argument('session', help='session: test, trainA, or trainB', nargs=1)
    parser.add_argument('--base_dir', dest = 'base_dir', help='Base directory to find files', default='/home/luke/Documents/Projects/RichardsPostdoc/Ensembles/CA1_imaging/')
    parser.add_argument('-r', '--redo', help='Redo source extraction')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()