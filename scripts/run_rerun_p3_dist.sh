#!/bin/bash

cd ~/DS-Analyzer/tool

for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
		for batch in 32 64 128 256; do
	    	echo "==============================================="
		    echo "$batch $arch"
		    echo "==============================================="
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=2 -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3-dist/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done

for arch in 'vgg11' 'resnet50'; do
		for batch in 32 64; do
	    	echo "==============================================="
		    echo "$batch $arch"
		    echo "==============================================="
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=2 -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3-dist/p3.8xlarge_2/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done

for arch in 'vgg11' 'resnet50'; do
		for batch in 80; do
	    	echo "==============================================="
		    echo "$batch $arch"
		    echo "==============================================="
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=2 -b $batch --steps RUN0 RUN1 RUN2 RUN3 --resume_dir results/results-p3-dist/p3.8xlarge_2/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py 
		done
done


<<'EOF'

for arch in 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'resnet18' 'resnet50' 'resnet34' 'resnet18' 'resnet101' 'resnet152'; do
		for batch in 32 48 64 80; do
	    echo "==============================================="
	    echo "$batch $arch"
	    echo "==============================================="
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=1 -b $batch --steps RUN0 RUN1 --resume_dir results/results-p3/p3.8xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
			python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=1 -b $batch --steps RUN0 RUN1 --resume_dir results/results-p3/p3.16xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

		done
done


for arch in 'resnet18' 'resnet50' 'resnet34' 'resnet18' 'resnet101' 'resnet152'; do
		for batch in 32 80; do
	    echo "==============================================="
	    echo "$batch $arch"
	    echo "==============================================="
			python3 -u harness.py --nproc_per_node=4 -j 16 -a $arch --nnodes=1 -b $batch --steps RUN0 RUN1 --resume_dir results/results-p3-synthetic/p3.8xlarge_noResidue/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 
#			python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=1 -b $batch --steps RUN0 RUN1 --resume_dir results/results-p3/p3.16xlarge/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

		done
done

EOF
