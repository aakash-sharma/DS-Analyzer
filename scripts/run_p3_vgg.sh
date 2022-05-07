#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
DIV_FACTOR=$5
NNODES=$6
MASTER=$7
NODE_RANK=$8
SAMPLES=1281166

cd ~/DS-Analyzer/tool
echo $NNODES
for arch in 'vgg11' 'vgg13' 'vgg16' 'vgg19'; do
	for batch in 80; do
		d=`date`
		echo "==============================================="
	    echo " $batch $arch $d"
	    echo "==============================================="

		if ((NNODES < 1)); then 
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --num_minibatches $((SAMPLES / batch / GPU / 2)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		else
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU / 2)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		fi
	done
done

