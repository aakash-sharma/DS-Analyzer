#!/bin/bash

NNODES=$1

cd ~/DS-Analyzer/tool

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
		for batch in 32 64 128 256 512; do
	    	echo "==============================================="
		    echo "$batch $arch"
		    echo "==============================================="
			python3 -u harness.py --nproc_per_node=1 -j 4 -a $arch --nnodes=$NNODES -b $batch --steps RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.2xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.8xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
			python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.16xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done


for arch in 'resnet50' 'vgg11'; do
		for batch in 32 64 128 256 512; do
	    echo "==============================================="
	    echo "$batch $arch"
	    echo "==============================================="
			python3 -u harness.py --nproc_per_node=1 -j 4 -a $arch --nnodes=$NNODES -b $batch --steps RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.2xlarge/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.8xlarge/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 
			python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=$NNODES -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3/p3.16xlarge/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 

		done
done

