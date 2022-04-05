python harness.py --nproc_per_node=$1 -j $2 -a resnet18 --nnodes=$3 --master_addr=$4 --master_port=11111 -b 256 --num_minibatches=86 --node_rank=$5 --prefix results/test image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval /home/ubuntu/ImageNet_Datasets

