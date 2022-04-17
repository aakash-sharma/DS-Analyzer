#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
NNODES=$5

cd ~/DS-Analyzer/tool

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
		for batch in 64 128 256 512; do
	    	echo "==============================================="
		    echo "$bacth $arch"
		    echo "==============================================="
			python3 -u harness.py --nproc_per_node=$GPU -j $CPU -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done


for arch in 'resnet50' 'vgg11'; do
		for batch in 64 128 256 512; do
	    echo "==============================================="
	    echo "$arch"
	    echo "==============================================="
		python3 -u harness.py --nproc_per_node=$GPU -j $CPU -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done

