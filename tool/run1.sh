#!/bin/bash


for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
         for batch in 256; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
             python -u harness.py --nproc_per_node=1 -j 2 -b $batch  -a $arch --prefix results/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
         done
done

for arch in 'resnet50' 'vgg11'; do
         for batch in 256; do
	     echo "==============================================="
	     echo " $batch $arch"
	     echo "==============================================="
             python harness.py --nproc_per_node=1 -j 2 -b $batch  -a $arch --prefix results/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /home/ubuntu/ImageNet_Datasets  >> ds_log 2>&1
         done
done
