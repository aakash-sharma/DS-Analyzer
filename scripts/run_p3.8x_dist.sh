#!/bin/bash

PREFIX=$1
INSTANCE=$2
DIV_FACTOR=$3
NNODES=$4
MASTER=$5
NODE_RANK=$6

SAMPLES=1281166
NUM_GPUS=$((4 * NNODES))

cd ~/DS-Analyzer/tool

for arch in 'alexnet' 'shufflenet_v2_x0_5' 'squeezenet1_0'; do
	d=`date`
	echo "==============================================="
	echo "256 $arch $d"
	echo "==============================================="
	python -u harness.py --nproc_per_node=4 -j 16 -b 256 -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 256 / NUM_GPUS)) --synthetic_div_factor $DIV_FACTOR --steps RUN0 RUN1 --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
done

for arch in 'mobilenet_v2' 'resnet18'; do
	d=`date`
	echo "==============================================="
	echo "128 $arch $d"
	echo "==============================================="
	python -u harness.py --nproc_per_node=4 -j 16 -b 128 -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 128 / NUM_GPUS)) --synthetic_div_factor $DIV_FACTOR --steps RUN0 RUN1 --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
done


for arch in 'resnet50' 'vgg11'; do
	d=`date`
	echo "==============================================="
	echo " 80 $arch $d"
	echo "==============================================="
	python harness.py --nproc_per_node=4 -j 16 -b 80  -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 80 / NUM_GPUS)) --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1
done

cd ${PREFIX}
tar -czvf ${INSTANCE}-rank${NODE_RANK}.tar.gz ${INSTANCE}


if [[ $NODE_RANK == 0 ]]; then
	exit
fi

scp -o "StrictHostKeyChecking no" ${INSTANCE}-rank${NODE_RANK}.tar.gz $MASTER:~/DS-Analyzer/tool/${PREFIX}/

'
