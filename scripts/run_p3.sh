#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
SAMPLES=1281166

cd /home/abs5688/DS-Analyzer/tool

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11'; do
         for batch in 32 64 128 256 512; do
				 echo "==============================================="
				 echo " $batch $arch"
				 echo "==============================================="
				 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done