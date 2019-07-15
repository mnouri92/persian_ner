#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. The firts parameter is the path of the input directory. The second parameter is the path of the output directory..."
    exit 1
fi

pip install -r requirement.txt

mkdir $2
mkdir "nsurl/tmp"

python script_mtl_evaluate.py $1 "nsurl/tmp"
python finalize_nsurl.py "nsurl/tmp" $2

rm -rf "nsurl/tmp"



