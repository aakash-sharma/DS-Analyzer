#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
NNODES=$5
MASTER=$6
NODE_RANK=$7
SAMPLES=1281166

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5'; do
         for batch in 128; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
		 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done

for arch in 'mobilenet_v2' 'squeezenet1_0'; do
         for batch in 64; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
		python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
	done
done

echo "==============================================="
echo " 64 resnet50"
echo "==============================================="
batch=64
python harness.py --nproc_per_node=$GPU -j $CPU -b 64  -a resnet50 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1


echo "==============================================="
echo " 32 vgg11"
echo "==============================================="
batch=64
python harness.py --nproc_per_node=$GPU -j $CPU -b 32  -a vgg11 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1

if [[ $NODE_RANK == 0 ]]; then
	exit
fi

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'resnet50' 'vgg11'; do
	scp -o "StrictHostKeyChecking no" -r ${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/rank-$NODE_RANK $MASTER:~/DS-Analyzer/tool/${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/
	scp -o "StrictHostKeyChecking no" -r ${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/rank-$NODE_RANK $MASTER:~/DS-Analyzer/tool/${PREFIX}/${INSTANCE}/dali-cpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/
