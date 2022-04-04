#!/bin/bash

range=$1
i=0

declare -A slaves=( ["slave0"]="172.31.65.5" )

while [ $i -lt $range ];
do
    #ssh -o "StrictHostKeyChecking no" ${slave}$i "/proj/scheduler-PG0/scripts/limitCPU.sh > /users/aakashsh/limit.log 2>&1 &"
	echo "${slaves[0]}"
    i=$((i+1))
done
