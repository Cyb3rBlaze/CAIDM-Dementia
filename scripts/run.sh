#!/bin/bash
# Automates jarvis cluster training and evaluation.
# $1 - experiment folder
# $2 - regex of which GPU hardware to use

# Regex for new TF 2.5 cluster: "rtx.*worker-[0,1]"

if [ -z $1 ] || [ -z $2 ] ; then
    echo "ArgumentError - Script requires two arguments: (DIRECTORY, GPU_TYPES)"
    exit
fi

# Generates bash files for each permutation
jarvis script -jmodels "/home/mmorelan/proj/dementia/train/$1/" -output_dir "/home/mmorelan/proj/dementia/train/$1/scripts/"

# Sends the generated bash files to the GPU cluster
jarvis cluster add -scripts "/home/mmorelan/proj/dementia/train/$1/scripts/*.sh" -workers "$2"
