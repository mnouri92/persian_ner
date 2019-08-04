#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. The firts parameter must be the path of the word embedding file."
    exit 1
fi

pip install -r requirement.txt

mkdir files
mkdir files/stl
mkdir files/stl/ner_bijankhan
mkdir files/stl/ner_bijankhan/data
mkdir files/stl/ner_bijankhan/model

if [ ! -f files/mtl/ner_bijankhan/data/train.data ]
then
    wget https://www.dropbox.com/s/hdqf7j9ftyoccml/train.data?dl=0
    mv train.data?dl=0 files/mtl/ner_bijankhan/data/train.data
fi

if [ ! -f files/mtl/ner_bijankhan/we.vec ]
then
    cp "$1" "files/mtl/ner_bijankhan/we.vec"
fi

python script_train.py "mtl" files/mtl/ner_bijankhan/ files/mtl/ner_armanperso/

