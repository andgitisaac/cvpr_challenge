from __future__ import print_function, division

import os
import argparse
import numpy as np
import skimage.io
from utils import get_file_paths, get_image, image_resize

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    dest='mode', help='"train"(With Mask) or "validation"(No Mask)',
                    required=True)
parser.add_argument('--image-path', type=str,
                    dest='image_path', help='directory of images',
                    required=True)
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='directory of output spliting images',
                    required=True)
args = parser.parse_args()

def split(content, filename, mode):
    M = content.shape[0] // 4
    N = content.shape[1] // 4
    splited = [content[x:x+M,y:y+N] for x in range(0,content.shape[0],M) for y in range(0,content.shape[1],N)]
    for idx, img in enumerate(splited):
        img = image_resize(img, size=(512, 512))
        fileID = filename.split('/')[-1].split('_')[0]
        postfix = 'sat.jpg' if mode == 'sat' else 'mask.png'
        output_name = '{}-{:02d}_{}'.format(fileID, idx, postfix)
        print(output_name)
        skimage.io.imsave(os.path.join(args.output_path, output_name), img)
    

def main():
    if args.mode == 'train':
        content_path, mask_path = get_file_paths(args.image_path)    
        content_path.sort()
        mask_path.sort()
        for path in content_path:
            split(get_image(path), path, mode='sat')
        for path in mask_path:
            split(get_image(path), path, mode='mask')
    elif args.mode == 'validation':
        content_path, _ = get_file_paths(args.image_path)    
        content_path.sort()
        for path in content_path:
            split(get_image(path), path, mode='sat')

if __name__ == '__main__':
    main()