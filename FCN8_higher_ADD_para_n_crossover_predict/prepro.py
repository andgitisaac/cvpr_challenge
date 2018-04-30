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
    splited = [content[x:x+M,y:y+N] for x in range(0,content.shape[0],M//4) if (x+M <= content.shape[0]) for y in range(0,content.shape[1],N//4) if (y+N <= content.shape[1])]
    for idx, img in enumerate(splited):
        img = image_resize(img, size=(512, 512))
        fileID = filename.split('/')[-1].split('_')[0]
        postfix = 'sat.jpg' if mode == 'sat' else 'mask.png'
        output_name = '{}-{:03d}_{}'.format(fileID, idx, postfix)
        print("\r{}".format(output_name), end='')
        skimage.io.imsave(os.path.join(args.output_path, output_name), img)
    print("")
    

def main():
    if args.mode == 'train':
        content_path, mask_path = get_file_paths(args.image_path)    
        content_path.sort()
        mask_path.sort()
        mask_size = list(map(os.path.getsize, mask_path))
        _, content_path = zip(*sorted(zip(mask_size, content_path), reverse=True))

        for path in content_path[20:80]:
            split(get_image(path), path, mode='sat')
        


    elif args.mode == 'validation':
        content_path, _ = get_file_paths(args.image_path)    
        content_path.sort()
        for idx, path in enumerate(content_path):
            print("{}/{}".format(idx, len(content_path)))
            split(get_image(path), path, mode='sat')

if __name__ == '__main__':
    main()