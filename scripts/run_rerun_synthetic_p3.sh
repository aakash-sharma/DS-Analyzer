#!/bin/bash

cd ~/DS-Analyzer/tool

<<'EOF'

EOF

for arch in 'resnet10' 'resnet12' 'resnet16' 'noResidue_resnet10' 'noResidue_resnet12' 'noResidue_resnet16' 'noResidue_resnet18' 'noResidue_resnet34' 'noResidue_resnet50' 'noResidue_resnet101' 'noResidue_resnet152' 'noBN_resnet10' 'noBN_resnet12' 'noBN_resnet16' 'noBN_resnet18' 'noBN_resnet34' 'noBN_resnet50' 'noBN_resnet101' 'noBN_resnet152'; do
		for repeat in 'repeat-0' 'repeat-1' 'repeat-2'; do
				echo "==============================================="
				echo "$repeat $arch"
				echo "==============================================="
					python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=1 -b 32 --steps RUN0 RUN1 --resume_dir results/results-p3-synthetic/p3.16xlarge/$repeat/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

		done
done

for arch in 'resnet10' 'resnet12' 'resnet16' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152' 'vgg11' 'vgg13' 'vgg16' 'vgg19'; do
    for repeat in 'repeat-0' 'repeat-1' 'repeat-2'; do
        d=`date`
        echo "==============================================="
        echo " $batch $arch $repeat $d"
        echo "==============================================="
					python3 -u harness.py --nproc_per_node=8 -j 32 -a $arch --nnodes=1 -b 32 --steps RUN0 RUN1 --resume_dir results/results-p3-synthetic/p3.16xlarge/$repeat/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py 

    done
done

