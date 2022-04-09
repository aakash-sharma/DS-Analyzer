#!/bin/bash

GPU=$1
CPU=$2
BATCH=$3
PREFIX=$4
INSTANCE=$5
SAMPLES=1281166/16

cd /home/ubuntu/DS-Analyzer/tool

python -u harness.py --nproc_per_node=$GPU -j $CPU -b $BATCH  -a shufflenet_v2_x0_5 --num_minibatches $((SAMPLES / BATCH / GPU / 2)) --steps RUN1 RUN2 RUN3 --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1

