#!/bin/bash

for file in `find . -name "*MODEL.json"`; do
		path=${file%/*}
		echo $path
		path="${path}/cpus-4/rank-0/MODEL.json"
		mv $file $path
done
