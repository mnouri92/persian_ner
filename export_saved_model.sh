#!/bin/bash

if [ "$#" -ne 3 ]
then
    echo "Please use the shell script correctly -> ( ./export_saved_model.sh <model_type(mtl2 , mtl3 or mtl4)> <choosed_task_path> <model_version_number> )"
    exit 1
fi


python export_saved_model.py $1 $2/model files/mtl/ner_bijankhan/we.vec $2/saved_model $3
