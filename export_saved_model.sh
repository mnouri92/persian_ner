#!/bin/bash

if [ "$#" -ne 2 ]
then
    echo "Please use the shell script correctly -> ( ./export_saved_model.sh <model_type(mtl2 , mtl3 or mtl4)> <choosed_task_path> )"
    exit 1
fi


python export_saved_model.py $1 $2/model files/mtl/ner_bijankhan/we.vec $2/saved_model
