#!/bin/bash

if [ "$#" -lt 1 ]
then
    echo "Illegal number of parameters. The firts parameter must be the choosed model to train ( mtl2, mtl3 or mtl4 )."
    exit 1
fi

if [ "$1" != "mtl2" ] && [ "$1" != "mtl3" ] && [ "$1" != "mtl4" ]
then

	echo "Please choose model type from [mtl2, mtl3 and mtl4]"
	exit 1
fi

pip install -r requirement.txt

mkdir -p files
mkdir -p files/mtl
mkdir -p files/mtl/ner_bijankhan
mkdir -p files/mtl/ner_bijankhan/data
mkdir -p files/mtl/ner_bijankhan/model
mkdir -p files/mtl/ner_bijankhan/saved_model
mkdir -p files/mtl/ner_armanperso
mkdir -p files/mtl/ner_armanperso/data
mkdir -p files/mtl/ner_armanperso/model
mkdir -p files/mtl/ner_armanperso/saved_model
mkdir -p files/mtl/gen
mkdir -p files/mtl/gen/data
mkdir -p files/mtl/gen/model
mkdir -p files/mtl/gen/saved_model
mkdir -p files/mtl/pos
mkdir -p files/mtl/pos/data
mkdir -p files/mtl/pos/model
mkdir -p files/mtl/pos/saved_model

if [ ! -f files/mtl/ner_bijankhan/data/train.data ]
then
    wget https://www.dropbox.com/s/hdqf7j9ftyoccml/train.data?dl=0
    mv train.data?dl=0 files/mtl/ner_bijankhan/data/train.data
fi

if [ ! -f files/mtl/ner_armanperso/data/train.data ]
then
    wget https://www.dropbox.com/s/ju6okh5aw24ozyy/train.data?dl=0
    mv train.data?dl=0 files/mtl/ner_armanperso/data/train.data
fi

if [ ! -f files/mtl/gen/data/train.data ]
then
    wget https://www.dropbox.com/s/qkujf49pf1r24kf/train.data?dl=0
    mv train.data?dl=0 files/mtl/gen/data/train.data
fi

if [ ! -f files/mtl/pos/data/train.data ]
then
    wget https://www.dropbox.com/s/y15iuw6ngxycd4h/train.data?dl=0
    mv train.data?dl=0 files/mtl/pos/data/train.data
fi

if [ ! -f files/mtl/ner_bijankhan/we.vec ]
then
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
    tar -xzf ./cc.fa.300.vec.tar.gz
    mv ./cc.fa.300.vec ./files/mtl/ner_bijankhan/we.vec
    rm ./cc.fa.300.vec.tar.gz
fi


if [ "$1" == "mtl2" ]
then
    if [ "$#" -ne 3 ]
    then
        echo "please use the shell correctly : ( ./train.sh mtl2 <main_task_path> <auxiliary_task_path> )"
        exit 1
    fi
    python script_train.py $1 $2/model files/mtl/ner_bijankhan/we.vec $2 $3
fi


if [ "$1" == "mtl3" ]
then
    if [ "$#" -ne 4 ]
    then
        echo "please use the shell correctly : ( ./train.sh mtl3 <main_task_path> <first_aux_task_path> <second_aux_task_path> )"
        exit 1
    fi
    python script_train.py $1 $2/model files/mtl/ner_bijankhan/we.vec $2 $3 $4
fi



if [ "$1" == "mtl4" ]
then
    if [ "$#" -ne 5 ]
    then
        echo "please use the shell correctly : ( ./train.sh mtl4 <main_task_path> <first_aux_task_path> <second_aus_task_path> <third_aux_task_path> )"
        exit 1
    fi
    python script_train.py $1 $2/model files/mtl/ner_bijankhan/we.vec $2 $3 $4 $5
fi
