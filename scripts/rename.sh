#!/bin/bash

root=~/DS-Analyzer/tool/results/results-p3-synthetic/p3.16xlarge_noResidue/
cd $root

<<'EOF'

EOF

for repeat in 'repeat-0' 'repeat-1' 'repeat-2'; do

		cd $repeat/dali-gpu

		rm -rf noBN_*
		rm -rf vgg*
		rm -rf resnet*


		#for file in noBN_*; do
		for file in noResidue_*; do

				new_file=${file//noResidue_}
				echo $file $new_file
				mv "$file" $new_file
		done

		cd $root
done
