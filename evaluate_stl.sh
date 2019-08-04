#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. The firts parameter is the path of the input directory. The second parameter is the path of the output directory..."
    exit 1
fi

pip install -r requirement.txt

mkdir $2

python script_evaluate.py "stl" "files/mtl/ner_bijankhan/" $1 $2


