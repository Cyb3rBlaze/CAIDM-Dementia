#!/bin/bash
# Automates Tensorboard deployment for a live cluster experiment.
# $1 - experiment folder

if [ -z $1 ] ; then
    echo "ArgumentError - Script requires one argument: (DIRECTORY)"
    exit
fi

# Generates bash files for each permutation
TENSORBOARD_CMD="tensorboard --logdir \"/home/${USER}/proj/dementia/train/$1/jmodels/logdirs/\" --bind_all --port 0"
echo $TENSORBOARD_CMD
read -p "Proceed with command (Y/n)? " RESP
if [ "$RESP" == "Y" ] || [ "$RESP" == "y" ] || [ "$RESP" == "" ]
then
    eval "$TENSORBOARD_CMD"
fi