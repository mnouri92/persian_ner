#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters. The first parameter is the type of model, second parameters is the path of the input directory, and The third parameter is the path of the output directory..."
    exit 1
fi

pip install -r requirement.txt

mkdir $3

python script_evaluate.py $1 "files/mtl/ner_bijankhan/" $2 $3


