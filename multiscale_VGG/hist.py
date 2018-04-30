from __future__ import print_function, division

from utils import get_file_paths, get_image
import numpy as  np

color2index = {
    (0  , 255, 255) : 0,
    (255, 255,   0) : 1,
    (255,   0, 255) : 2,
    (0  , 255,   0) : 3,
    (  0,   0, 255) : 4,
    (255, 255, 255) : 5,
    (  0,   0,   0) : 6
}

hist = [0] * 7
_, mask_path = get_file_paths('dataset/train/')
for ii, path in enumerate(mask_path):
    img = get_image(path)
    label = np.ndarray(shape=img.shape[:2], dtype=np.uint8)
    label[:, :] = -1
    for rgb, idx in color2index.items():
        label[(img == rgb).all(2)] = idx

    unique, counts = np.unique(label, return_counts=True)
    for i, class_ in enumerate(unique):
        if class_ != -1:
            hist[class_] += counts[i]
        else:
            print("\n-1 is in {}!".format(path))
    print('\r({}/{}) hist:{}'.format(ii+1, len(mask_path), hist), end='')

ratio = np.array(hist)
ratio = ratio / sum(ratio)
print('ratio:{}'.format(ratio))


    

