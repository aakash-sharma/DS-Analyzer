#!/bin/bash

GPU=$1
CPU=$2
NUM_NODES=$3
MASTER=$4
NODE_RANK=$5

source activate hpcl_dl
cd /home/ubuntu/DS-Analyzer/tool

python harness.py --nproc_per_node=$1 -j $2 -a resnet18 --nnodes=$3 --master_addr=$4 --master_port=11111 -b 256 --num_minibatches=86 --steps RUN1 RUN2 --node_rank=$5 --prefix results/test image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval /home/ubuntu/ImageNet_Datasets

if [[ $NODE_RANK == 0 ]]; then
        exit
fi

GPUS=$((GPU*NUM_NODES))

scp -o "StrictHostKeyChecking no" -r results/test/resnet18/jobs-1/gpus-$GPUS/cpus-$CPU/rank-$NODE_RANK $MASTER:~/DS-Analyzer/tool/results/test/resnet18/jobs-1/gpus-$GPUS/cpus-$CPU/


