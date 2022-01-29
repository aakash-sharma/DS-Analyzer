#!/bin/bash

GPU=$1
CPU=$2
SAMPLES=$3
batch=$((SAMPLES / 256))

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2'; do
         for batch in 256; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
		 python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch  -a $arch --num_minibatches $((SAMPLES / 256)) --prefix results/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done

echo "==============================================="
echo " 128 squeezenet1_0"
echo "==============================================="
python -u harness.py --nproc_per_node=$GPU -j $CPU -b 128  -a squeezenet1_0 --num_minibatches $((SAMPLES / 128)) --prefix results/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1

echo "==============================================="
echo " 64 resnet50"
echo "==============================================="
python harness.py --nproc_per_node=$GPU -j $CPU -b 128  -a resnet50 --num_minibatches $((SAMPLES / 64)) --prefix results/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1


echo "==============================================="
echo " 32 vgg11"
echo "==============================================="
python harness.py --nproc_per_node=$GPU -j $CPU -b 128  -a vgg11 --num_minibatches $((SAMPLES / 32)) --prefix results/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1
