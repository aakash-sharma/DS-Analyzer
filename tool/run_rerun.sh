#!/bin/bash

GPU=$1
CPU=$2
NNODES=$3
RESULT_DIR=$4
INSTANCE=$5

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
	     echo "==============================================="
	     echo "$arch"
	     echo "==============================================="
		 python3 -u harness.py --nproc_per_node=$GPU -j $CPU -a $arch --nnodes=$NNODES --steps RUN1 RUN2 RUN3 --resume_dir ${RESULT_DIR}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
done

for arch in 'resnet50' 'vgg11'; do
	     echo "==============================================="
	     echo "$arch"
	     echo "==============================================="
		 python3 -u harness.py --nproc_per_node=$GPU -j $CPU -a $arch --nnodes=$NNODES --steps RUN1 RUN2 RUN3 --resume_dir ${RESULT_DIR}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 
done

