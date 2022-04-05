#!/bin/bash

num_slaves=$1
num_gpu=$2
master=$3
num_cpu=$((num_gpu*2))
i=0

declare -A slaves=( ["slave0"]="172.31.65.5" 
					["slave1"]="172.31.65.6" 
					["slave2"]="172.31.65.12" 
					["slave3"]="172.31.65.8" 
					["slave4"]="172.31.65.9" 
					["slave5"]="172.31.65.10" 
					["slave6"]="172.31.65.11"
					["slave7"]="172.31.65.13"
					["slave8"]="172.31.65.14"
					["slave9"]="172.31.65.15"
					["slave10"]="172.31.65.16"
					["slave11"]="172.31.65.17"
					["slave12"]="172.31.65.18"
					["slave13"]="172.31.65.19"
					["slave14"]="172.31.65.20")

while [ $i -lt $num_slaves ];
do
		ssh -o "StrictHostKeyChecking no" ${slaves["slave$i"]} "~/DS-Analyzer/tool/run_test_dist.sh $num_gpu $num_cpu $((num_slaves+1)) $master > ~/ds.log 2>&1 &"
	echo ${slaves["slave$i"]}
    i=$((i+1))
done
