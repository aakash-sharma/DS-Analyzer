#!/bin/bash

cd ~/DS-Analyzer/tool

NNODES=2

python3 -u harness.py --nproc_per_node=4 -j 16 -a alexnet --nnodes=$NNODES -b 256 --resume_dir results/results-p3/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

python3 -u harness.py --nproc_per_node=4 -j 16 -a resnet18 --nnodes=$NNODES -b 128 --resume_dir results/results-p3/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

python3 -u harness.py --nproc_per_node=4 -j 16 -a shufflenet_v2_x0_5 --nnodes=$NNODES -b 256 --resume_dir results/results-p3/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

python3 -u harness.py --nproc_per_node=4 -j 16 -a squeezenet1_0 --nnodes=$NNODES -b 256 --resume_dir results/results-p3/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 



#for arch in 'resnet50' 'vgg11'; do

