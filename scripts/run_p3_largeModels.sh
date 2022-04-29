#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
DIV_FACTOR=$5
SAMPLES=1281166

cd ~/DS-Analyzer/tool


for arch in 'vgg11' 'resnet50'; do
         for batch in 80; do
                                 d=`date`
				 echo "==============================================="
				 echo "$d $batch $arch"
				 echo "==============================================="
				 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / batch / GPU )) --steps RUN2 RUN3 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done

#for arch in 'vgg11'; do
#         for batch in 64 128 256 512; do
#                                 d=`date`
#				 echo "==============================================="
#				 echo "$d $batch $arch"
#				 echo "==============================================="
#				 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / batch / GPU )) --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
#         done
#done


cp ds_log ${PREFIX}/${INSTANCE}
