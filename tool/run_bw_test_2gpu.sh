
CUDA_VISIBLE_DEVICES=0 ./bw_test > gpu0.out &
CUDA_VISIBLE_DEVICES=1 ./bw_test > gpu1.out 

sleep 60

HOST_TO_DEVICE=0
DEVICE_TO_HOST=0

for filename in gpu*.out; do
	HTD=$(cat $filename | grep "Host to Device bandwidth" |  awk '{print $(NF)}')
	DTH=$(cat $filename | grep "Device to Host bandwidth" |  awk '{print $(NF)}')
	HOST_TO_DEVICE=`echo "$HOST_TO_DEVICE+$HTD" | bc`
	DEVICE_TO_HOST=`echo "$DEVICE_TO_HOST+$HTD" | bc`
done

HOST_TO_DEVICE=`echo "$HOST_TO_DEVICE / 2" | bc -l`
DEVICE_TO_HOST=`echo "$DEVICE_TO_HOST / 2" | bc -l`
echo "Host to device bandwith (GB/s): $HOST_TO_DEVICE"
echo "Device to Host bandwidth (GB/s): $DEVICE_TO_HOST"
