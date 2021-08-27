#!/bin/bash
# Automates Tensorboard deployment for a live cluster experiment
# $1 - experiment folder

if [ -z $1 ] ; then
    echo "ArgumentError - Script requires one argument: (DIRECTORY)"
    exit
fi

tensorboard --logdir /home/${USER}/proj/dementia/train/$1/jmodels/logdirs/ --bind_all --port 0
