#!/bin/bash

#root=~/DS-Analyzer/tool/results/results-p3-synthetic/p3.16xlarge_noResidue/
root=~/DS-Analyzer/tool/results/results-p3-synthetic/results-p3-synthetic-resnet-noBn
cd $root

<<'EOF'

EOF

for instance in 'p3.16xlarge' 'p3.8xlarge_2'; do

		for repeat in 'repeat-0' 'repeat-1' 'repeat-2'; do

				cd $instance/$repeat/dali-gpu

				#rm -rf noBN_*
				rm -rf noRes_*
		
				rm -rf vgg*
				rm -rf resnet*


				for file in noBN_*; do
				#for file in noResidue_*; do

						#new_file=${file//noResidue_}
						new_file=${file//noBN_}
						echo $file $new_file
						mv "$file" $new_file
				done

				cd $root
		done
done
