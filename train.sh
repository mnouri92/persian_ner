#!/bin/bash

if [ ! -d persian_ner ]; then
  git clone https://github.com/hadibokaei/persian_ner.git
  mkdir persian_ner/files
  mkdir persian_ner/files/data
  mkdir persian_ner/files/model
  mkdir persian_ner/files/data/ner_bijankhan
  mkdir persian_ner/files/data/ner_armanperso
fi
cd persian_ner
git pull


wget -c https://www.dropbox.com/s/ju6okh5aw24ozyy/train.data?dl=0
mv train.data?dl=0 persian_ner/files/data/ner_bijankhan/train.data

wget -c https://www.dropbox.com/s/4fjosovc8e10wro/validation.data?dl=0
mv train.data?dl=0 persian_ner/files/data/ner_bijankhan/validation.data

wget -c https://www.dropbox.com/s/hdqf7j9ftyoccml/train.data?dl=0
mv train.data?dl=0 persian_ner/files/data/ner_armanperso/train.data

cd persian_ner

python script_mtl_train.py
