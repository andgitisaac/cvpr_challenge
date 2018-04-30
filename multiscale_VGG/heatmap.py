from __future__ import print_function, division

import os
import argparse
import numpy as np
import seaborn
import matplotlib.pylab as plt
from utils import get_file_paths, get_image, mask_preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--predict-path', type=str,
                    dest='predict_path', help='directory of predicted masks',
                    required=True)
parser.add_argument('--gt-path', type=str,
                    dest='gt_path', help='directory of GT masks',
                    required=True)
args = parser.parse_args()

def main():
    _, predict_path = get_file_paths(args.predict_path)

    error = np.zeros((7, 7))
    for idx, path in enumerate(predict_path):        
        fileID = path.split('/')[-1].split('_')[0] + '_mask.png'
        gt_path = os.path.join(args.gt_path, fileID)

        p_mask = mask_preprocess(get_image(path)).astype(int)   
        g_mask = mask_preprocess(get_image(gt_path)).astype(int) 
        
        sub = p_mask - g_mask
        height, width, _ = sub.shape

        for h in range(height):
            for w in range(width):
                if not np.any(sub[w, h, :]): # all zeros in this pixel (predict = gt)
                    gt = np.where(g_mask[w, h, :]==1)[0][0]
                    error[gt, gt] = error[gt, gt] + 1  
                else:
                    gt = np.where(sub[w, h, :]==-1)[0][0]
                    p = np.where(sub[w, h, :]==1)[0][0]
                    error[gt, p] = error[gt, p] + 1      

            print("({}/{} {:03d}%) processing".format(idx+1, len(predict_path), int(100*(h*height+w)/height/width)))
        np.save('error', error)
        print('saved')
    np.save('error', error)
    print('saved')
    # error = np.load('error.npy')

    for i in range(7):
        if not np.any(error[i, :]):
            continue
        error[i, :] = error[i, :] / sum(error[i, :])

    vmin = np.min(error[:])
    vmax = np.max(error[:])

    class_ = ['Urban land', 'Agriculture land', 'Rangeland', 'Forest land', 'Water', 'Barren land', 'Unknown']
    fig = plt.figure(figsize=(18, 18))
    ax = seaborn.heatmap(error, linewidth=0.5, cmap='YlGn', annot=True, fmt=".03f", xticklabels=class_, yticklabels=class_, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Predicted Class', fontsize=30, labelpad=40)
    ax.set_ylabel('Actual Class', fontsize=30)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=0)
    # plt.show()
    plt.savefig('heatmap.png')
    




if __name__ == '__main__':
    main()