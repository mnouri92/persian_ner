#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. The firts parameter must be the path of the word embedding file."
    exit 1
fi

pip install -r requirement.txt

mkdir files
mkdir files/mtl
mkdir files/mtl/ner_bijankhan
mkdir files/mtl/ner_bijankhan/data
mkdir files/mtl/ner_bijankhan/model
mkdir files/mtl/ner_armanperso
mkdir files/mtl/ner_armanperso/data

if [ ! -f files/mtl/ner_bijankhan/data/train.data ]
then
    wget https://www.dropbox.com/s/hdqf7j9ftyoccml/train.data?dl=0
    mv train.data?dl=0 files/mtl/ner_bijankhan/data/train.data
fi

if [ ! -f files/we.vec ]
then
    cp $1 files/we.vec
fi

python script_train.py "stl" files/mtl/ner_bijankhan/
