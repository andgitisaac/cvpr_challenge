from __future__ import print_function, division

import os
import glob
import argparse
import numpy as np
from PIL import Image
from utils import get_file_paths, get_image, mask_postprocess

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str,
                    dest='image_path', help='directory of splited images',
                    required=True)
parser.add_argument('--output-path', type=str,
                    dest='output_path', help='directory of output whole images',
                    required=True)
args = parser.parse_args()


# def merge(group, size=(612, 612), num_per_side=4):
#     img = Image.new('RGB', (size[0]*num_per_side, size[1]*num_per_side))
#     for idx, path in enumerate(group):
#         offset_x = idx % num_per_side * size[0]
#         offset_y = idx // num_per_side * size[1]

#         chunk = Image.open(path)
#         img.paste(chunk, (offset_x, offset_y, offset_x+size[0], offset_y+size[1]))
#     return img

def merge(group, size=(612, 612), num_per_side=16):
    np_mask = np.zeros((2448, 2448, 7))
    img = Image.new('RGB', (2448, 2448))
    for idx, path in enumerate(group):
        offset_x = idx % num_per_side * size[0]
        offset_y = idx // num_per_side * size[1]

        chunk = np.load(path)
        np_mask[offset_x:offset_x+size[0], offset_y:offset_y+size[1], :] += chunk
    img = mask_postprocess(np_mask)
    return img

        
def main():
    mask_path = glob.glob('{}*.npz'.format(args.image_path))
    mask_path.sort()

    for i in range(len(mask_path) // 256):        
        group = mask_path[256*i:256*(i+1)]
        whole_img = merge(group)

        fileID = group[0].split('/')[-1].split('-')[0]
        output_name = '{}_mask.png'.format(fileID)
        print(output_name)

        whole_img.save(os.path.join(args.output_path, output_name))





if __name__ == '__main__':
    main()