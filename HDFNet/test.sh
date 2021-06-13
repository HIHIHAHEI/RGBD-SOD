#!/usr/bin/env bash

python test.py --param_path output/HDFNet_VGG19_7Datasets/pth/state_final.pth \
               --model HDFNet_VGG19 \
               --testset dataset/NLPR/test_data/ \
               --has_masks True \
               --save_pre True \
               --save_path output/HDFNet_VGG19_7Datasets/pre/test \
               --data_mode RGBD \
               --use_gpu True
