#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. The firts parameter is the path of the input directory. The second parameter is the path of the output directory..."
    exit 1
fi

mkdir $2
mkdir files
mkdir files/data
mkdir files/model
mkdir files/data/ner_bijankhan
mkdir files/data/ner_armanperso

mkdir "nsurl/tmp"

python script_mtl_evaluate3.py $1 "nsurl/tmp"
python finalize_nsurl.py "nsurl/tmp" $2

rm -rf "nsurl/tmp"



