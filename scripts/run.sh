#!/bin/bash
# Automates jarvis cluster training and evaluation.
# $1 - experiment folder
# $2 - regex of which GPU hardware to use (rtx / gtx)

# Regex for new TF 2.5 cluster: "rtx.*worker-[0,1]"

if [ -z $1 ] || [ -z $2 ] ; then
    echo "ArgumentError - Script requires two arguments: (DIRECTORY, GPU_TYPES)"
    exit
fi

# Generates bash files for each permutation
SCRIPT_CMD="jarvis script -jmodels \"/home/${USER}/proj/dementia/train/$1/\" -output_dir \"/home/${USER}/proj/dementia/train/$1/scripts/\""
echo $SCRIPT_CMD
read -p "Proceed with command (Y/n)? " RESP
if [ "$RESP" == "Y" ] || [ "$RESP" == "y" ] || [ "$RESP" == "" ]
then
    eval "$SCRIPT_CMD"
fi

# Sends the generated bash files to the GPU cluster
CLUSTER_CMD="jarvis cluster add -scripts \"/home/${USER}/proj/dementia/train/$1/scripts/*.sh\" -workers \"$2.*worker-[0,1]\""
echo $CLUSTER_CMD
read -p "Proceed with command (Y/n)? " RESP
if [ "$RESP" == "Y" ] || [ "$RESP" == "y" ] || [ "$RESP" == "" ]
then
    eval "$CLUSTER_CMD"
fi

