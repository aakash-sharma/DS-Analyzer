#!/bin/bash

GPU=$1
CPU=$2
PREFIX=$3
INSTANCE=$4
DIV_FACTOR=$5
NNODES=$6
MASTER=$7
NODE_RANK=$8
SAMPLES=1281166
NUM_GPUS=$((GPU * NNODES))

echo $NUM_GPUS

cd ~/DS-Analyzer/tool


#for arch in 'resnet50' 'vgg11'; do
#	for batch in 80; do
#	    d=`date`
#	    echo "==============================================="
#	    echo " $batch $arch $d"
#	    echo "==============================================="
#		python -u harness.py --nproc_per_node=$GPU -j $CPU -b $batch -a $arch --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((SAMPLES / batch / NUM_GPUS)) --steps RUN1 RUN2 RUN3 --synthetic_div_factor $DIV_FACTOR --prefix ${PREFIX}/${INSTANCE}/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /home/ubuntu/ImageNet_Datasets >> ds_log 2>&1
#	done
#done

python harness.py --nproc_per_node=$GPU -j $CPU -a BERT -b 4 --nnodes=$NNODES --master_addr=$MASTER --master_port=11111 --node_rank=$NODE_RANK --num_minibatches $((5424 / NUM_GPUS)) --synthetic_div_factor 1 --steps  RUN2 RUN3 --prefix ${PREFIX}/${INSTANCE} BERT/bert_squad_dist_stash.py >> ds_log 2>&1

cd ${PREFIX}
tar -czvf ${INSTANCE}-rank${NODE_RANK}.tar.gz ${INSTANCE}


if [[ $NODE_RANK == 0 ]]; then
    exit
fi

scp -o "StrictHostKeyChecking no" ${INSTANCE}-rank${NODE_RANK}.tar.gz $MASTER:~/DS-Analyzer/tool/${PREFIX}/
