#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
DIV_FACTOR=$5
DATA_DIR=$6
SAMPLES=1281166

python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / batch / GPU )) --synthetic_div_factor $DIV_FACTOR --steps RUN3 --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data ${DATA_DIR}
