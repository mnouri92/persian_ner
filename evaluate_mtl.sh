#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters. The first parameter is the type of model, second parameters is the model path, and the third and forth parameters are the input and output data path respectively"
    exit 1
fi

pip install -r requirement.txt

mkdir $4

python script_evaluate.py $1 $2 "files/mtl/ner_bijankhan/" $3 $4


