#!/bin/bash

echo $1

mkdir files
mkdir files/data
mkdir files/model
mkdir files/data/ner_bijankhan
mkdir files/data/ner_armanperso

git pull

cp $1 ./
rm -rf nsurl
mkdir nsurl
unzip tokenout.zip -d nsurl

mkdir "nsurl/results"
mkdir "nsurl/final_results"

python script_mtl_evaluate3.py "nsurl/tokenout" "nsurl/results/"
python finalize_nsurl.py nsurl/results nsurl/final_results

cd "nsurl/final_results"
zip prediction.zip ./*



