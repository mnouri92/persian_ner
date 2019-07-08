#!/bin/bash

pip install hazm
pip install numpy==1.16.1
pip install tensorflow==1.14

mkdir files
mkdir files/data
mkdir files/model
mkdir files/data/ner_bijankhan
mkdir files/data/ner_armanperso

git pull

if [ ! -f files/data/ner_bijankhan/train.data ]
then
    wget https://www.dropbox.com/s/hdqf7j9ftyoccml/train.data?dl=0
    mv train.data?dl=0 files/data/ner_bijankhan/train.data
fi

if [ ! -f files/data/ner_bijankhan/validation.data ]
then
    wget https://www.dropbox.com/s/4fjosovc8e10wro/validation.data?dl=0
    mv validation.data?dl=0 files/data/ner_bijankhan/validation.data
fi

if [ ! -f files/data/ner_armanperso/train.data ]
then
    wget https://www.dropbox.com/s/ju6okh5aw24ozyy/train.data?dl=0
    mv train.data?dl=0 files/data/ner_armanperso/train.data
fi

python script_mtl_train.py
