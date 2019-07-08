#!/bin/bash

echo $1

#pip install hazm
#pip install numpy==1.16.1
#pip install tensorflow==1.14

mkdir files
mkdir files/data
mkdir files/model
mkdir files/data/ner_bijankhan
mkdir files/data/ner_armanperso

git pull

cp $1 tokenout.zip
rm -rf nsurl
mkdir nsurl
unzip tokenout.zip -d nsurl

mkdir "nsurl/results"
mkdir "nsurl/final_results"

python script_mtl_evaluate3.py "nsurl/" "nsurl/results/"
python finalize_nsurl nsurl/results nsurl/final_results

cd "nsurl/final_results"
zip prediction.zip ./*



