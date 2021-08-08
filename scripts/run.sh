#!/bin/bash
# Automates jarvis cluster training and evaluation.
# $1 - experiment folder
# $2 - regex of which GPU hardware to use

if [ -z $1 ] || [ -z $2 ] ; then
    echo "ArgumentError - Script requires two arguments: (DIRECTORY, GPU_TYPES)"
    exit
fi

path=$(echo $(cd ../ && pwd)/"train")

# Set python path
python $path/$1/"env.py" $path/$1/"train.py"

# # Generates bash files for each permutation
jarvis script -py "train" -jmodels $path/$1/ -output_dir $path/$1/"scripts"/

# # Sends the generated bash files to the GPU cluster
jarvis cluster add -scripts $path/$1/"scripts/*.sh" -workers "$2"