#!/bin/bash

input_dir=../data_raw #Dataset files

output_dir=../result #output data dir, all output will be here
nums_files=30 #numbers of the whole dataset

use_active=True
stage=0

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
	#Delete folder "audio" and "ext" because each time run create_inital_mixtures.py
       	#  will generate new wav pairs for mixtures, so we need to remove the old folders.
	#  (This process can be done in python if necessary)

	
	python create_inital_mixtures.py --input_dir $input_dir --output_dir $output_dir --nums_files $nums_files
fi

if [ $stage -le 2 ];then

	python create_mixtures.py --data_dir $output_dir --state train --use_active $use_active
	python create_mixtures.py --data_dir $output_dir --state test --use_active $use_active
fi

if [ $stage -le 3 ];then

	python create_good_scp.py --data_dir $output_dir

fi

#TODO
#Will update STFT, CMVN as soon as possible.
