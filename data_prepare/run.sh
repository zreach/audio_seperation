#!/bin/bash


MAIN_ROOT=$PWD/..
SRC_ROOT=$MAIN_ROOT/data_prepare
export PATH=$SRC_ROOT/:$PATH
# export PYTHONPATH=$SRC_ROOT/:$PYTHONPATH

# export PATH=$PATH/data_prepare

input_dir=./data_raw #Dataset files

output_dir=../result #output data dir, all output will be here
nums_files=30 #numbers of the whole dataset

use_active=True
stage=1

if [ ! -d $output_dir ];then
	mkdir $output_dir
else
	echo "$output_dir existed"
fi

if [ $stage -le 1 ];then

	if [ -d $output_dir"/""audio" ];then
		rm -rf $output_dir"/""audio"
	fi
	if [ -d $output_dir"/""text" ];then
		rm -rf $output_dir"/""text"
	fi
	
	python create_initial_mixtures.py --input_dir $input_dir --output_dir $output_dir --nums_files $nums_files
fi

if [ $stage -le 2 ];then

	python create_mixtures.py --data_dir $output_dir --state train --use_active $use_active
	python create_mixtures.py --data_dir $output_dir --state test --use_active $use_active
fi