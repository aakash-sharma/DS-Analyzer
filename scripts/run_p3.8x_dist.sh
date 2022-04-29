#!/bin/bash

NNODES=$1
MASTER=$2
NODE_RANK=$3
PREFIX=$4
INSTANCE=$5

SAMPLES=1281166
NUM_GPUS=((4 * NNODES))

cd ~/DS-Analyzer/tool

d=`date`
echo "==============================================="
echo "256 alexnet $d"
echo "==============================================="
python -u harness.py --nproc_per_node=4 -j 16 -b 256 -a alexnet --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 256 / NUM_GPUS)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1

d=`date`
echo "==============================================="
echo "128 resnet18 $d"
echo "==============================================="
python -u harness.py --nproc_per_node=4 -j 16 -b 128 -a resnet18 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 128 / NUM_GPUS)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1

d=`date`
echo "==============================================="
echo "256 shufflenet_v2_x0_5 $d"
echo "==============================================="
python -u harness.py --nproc_per_node=4 -j 16 -b 256 -a shufflenet_v2_x0_5 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 256 / NUM_GPUS)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1

d=`date`
echo "==============================================="
echo "256 squeezenet1_0 $d"
echo "==============================================="
python -u harness.py --nproc_per_node=4 -j 16 -b 256 -a squeezenet1_0 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / 256 / NUM_GPUS)) --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1





echo "==============================================="
echo " 64 resnet50"
echo "==============================================="
batch=64
#python harness.py --nproc_per_node=$GPU -j $CPU -b 64  -a resnet50 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1


echo "==============================================="
echo " 32 vgg11"
echo "==============================================="
batch=64
#python harness.py --nproc_per_node=$GPU -j $CPU -b 32  -a vgg11 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1

if [[ $NODE_RANK == 0 ]]; then
	exit
fi

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'resnet50' 'vgg11'; do
	scp -o "StrictHostKeyChecking no" -r ${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/rank-$NODE_RANK $MASTER:~/DS-Analyzer/tool/${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/
	scp -o "StrictHostKeyChecking no" -r ${PREFIX}/${INSTANCE}/dali-gpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/rank-$NODE_RANK $MASTER:~/DS-Analyzer/tool/${PREFIX}/${INSTANCE}/dali-cpu/$arch/jobs-1/gpus-$GPU/cpus-$CPU/
