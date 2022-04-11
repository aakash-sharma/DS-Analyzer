
CUDA_VISIBLE_DEVICES=0 ./bw_test > gpu0.out &
CUDA_VISIBLE_DEVICES=1 ./bw_test > gpu1.out &
CUDA_VISIBLE_DEVICES=2 ./bw_test > gpu2.out &
CUDA_VISIBLE_DEVICES=3 ./bw_test > gpu3.out &
CUDA_VISIBLE_DEVICES=4 ./bw_test > gpu4.out &
CUDA_VISIBLE_DEVICES=5 ./bw_test > gpu5.out &
CUDA_VISIBLE_DEVICES=6 ./bw_test > gpu6.out &
CUDA_VISIBLE_DEVICES=7 ./bw_test > gpu7.out &
CUDA_VISIBLE_DEVICES=8 ./bw_test > gpu8.out &
CUDA_VISIBLE_DEVICES=9 ./bw_test > gpu9.out &
CUDA_VISIBLE_DEVICES=10 ./bw_test > gpu10.out &
CUDA_VISIBLE_DEVICES=11 ./bw_test > gpu11.out &
CUDA_VISIBLE_DEVICES=12 ./bw_test > gpu12.out &
CUDA_VISIBLE_DEVICES=13 ./bw_test > gpu13.out &
CUDA_VISIBLE_DEVICES=14 ./bw_test > gpu14.out &
CUDA_VISIBLE_DEVICES=15 ./bw_test > gpu15.out 


sleep 180 

HOST_TO_DEVICE=0
DEVICE_TO_HOST=0

for filename in gpu*.out; do
        HTD=$(cat $filename | grep "Host to Device bandwidth" |  awk '{print $(NF)}')
        DTH=$(cat $filename | grep "Device to Host bandwidth" |  awk '{print $(NF)}')
        HOST_TO_DEVICE=`echo "$HOST_TO_DEVICE+$HTD" | bc`
        DEVICE_TO_HOST=`echo "$DEVICE_TO_HOST+$DTH" | bc`
done

HOST_TO_DEVICE=`echo "$HOST_TO_DEVICE / 16" | bc -l`
DEVICE_TO_HOST=`echo "$DEVICE_TO_HOST / 16" | bc -l`
echo "Host to device bandwith (GB/s): $HOST_TO_DEVICE"
echo "Device to Host bandwidth (GB/s): $DEVICE_TO_HOST"

