#!/bin/bash

export JARVIS_PATH_CONFIGS=/home/mmorelan/.jarvis
export JARVIS_PARAMS_CSV=/home/mmorelan/proj/dementia/train/96x128/hyper.csv
export JARVIS_PARAMS_ROW=0
python /home/mmorelan/proj/dementia/train/96x128/model.py > /home/mmorelan/proj/dementia/train/96x128/02/01/stdout 2>&1