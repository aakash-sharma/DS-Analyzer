#python harness.py --nproc_per_node=1 -j 1 -a resnet18 --prefix results/run1/ image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval /home/ubuntu/ImageNet_Datasets
python harness.py --nproc_per_node=1 -j 1 -a resnet18 --nnodes=2 --master_addr="172.31.75.25" --master_port=11111 -b 256 --num_minibatches=86 --node_rank=0 --prefix results/run1/ image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval /home/ubuntu/ImageNet_Datasets

