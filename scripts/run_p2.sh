#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
DIV_FACTOR=$5
SAMPLES=1281166

cd /home/ubuntu/DS-Analyzer/tool

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
         for batch in 32 64 96 128; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
		 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / batch / GPU)) --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done


<<'EOF'

echo "==============================================="
echo " 64 resnet50"
echo "==============================================="
batch=64
python harness.py --nproc_per_node=$GPU -j $CPU -b 64  -a resnet50 --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1


echo "==============================================="
echo " 32 vgg11"
echo "==============================================="
batch=64
python harness.py --nproc_per_node=$GPU -j $CPU -b 32  -a vgg11 --num_minibatches $((SAMPLES / batch / GPU / 2)) --prefix ${PREFIX}/${INSTANCE}/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1




EOF
