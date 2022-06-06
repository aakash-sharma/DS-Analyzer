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



for arch in 'resnet10' 'resnet12' 'resnet16' 'noResidue_resnet10' 'noResidue_resnet12' 'noResidue_resnet16' 'noResidue_resnet18' 'noResidue_resnet34' 'noResidue_resnet50' 'noResidue_resnet101' 'noResidue_resnet152' 'noBN_resnet10' 'noBN_resnet12' 'noBN_resnet16' 'noBN_resnet18' 'noBN_resnet34' 'noBN_resnet50' 'noBN_resnet101' 'noBN_resnet152'; do
	for repeat in 0 1 2; do
		d=`date`
		echo "==============================================="
	    echo " $batch $arch $repeat $d"
	    echo "==============================================="

		if ((NNODES < 2)); then 
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --num_minibatches $((SAMPLES / batch / GPU)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/repeat-{$repeat}/dali-gpu/  synthetic/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		else
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/repeat-{$repeat}/dali-gpu/  synthetic/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		fi
	done
done


for arch in 'resnet34' 'resnet101' 'resnet152' 'vgg13' 'vgg16' 'vgg19'; do
	for repeat in 0 1 2; do
		d=`date`
		echo "==============================================="
	    echo " $batch $arch $repeat $d"
	    echo "==============================================="

		if ((NNODES < 2)); then 
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --num_minibatches $((SAMPLES / batch / GPU)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/repeat-{$repeat}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		else
			python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU)) --steps RUN0 RUN1 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/repeat-{$repeat}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
		fi
	done
done
