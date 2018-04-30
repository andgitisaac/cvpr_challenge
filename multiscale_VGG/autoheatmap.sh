#!/bin/bash

#python3 -u /home/huaijing/cvpr_challenge/FCN8_higher_ADD_para_n_crossover_predict/main.py --mode eval --output-mode npz --model-path /home/huaijing/cvpr_challenge/FCN8_higher_ADD_para_n_crossover_predict/model/VGG_FCN8.44.h5 --test-path /home/huaijing/cvpr_challenge/FCN8_higher_ADD_para_n_crossover_predict/partial_train/splited/ --output-path /home/huaijing/cvpr_challenge/FCN8_higher_ADD_para_n_crossover_predict/partial_train/whole/
python3 -u heatmap.py --predict-path tmp/ --gt-path ../FCN8_higher_ADD_para_n_crossover_predict/dataset/train
