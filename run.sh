#!/usr/bin/env bash
cd src


if [ $1 -eq 0 ]
then
    export CUDA_VISIBLE_DEVICES=0
    python tf_luncher.py --rom_name breakout.bin --cut_top 40 --double_dqn --net_type 2 --data_dir ../data/networks0/ --log_file ../data/networks0/logs
elif [ $1 -eq 1 ]
then
    export CUDA_VISIBLE_DEVICES=1
    python tf_luncher.py --rom_name breakout.bin --cut_top 40 --data_dir ../data/networks1/ --log_file ../data/networks1/logs
elif [ $1 -eq 2 ]
then
    export CUDA_VISIBLE_DEVICES=2
    python tf_luncher.py --rom_name seaquest.bin --cut_top 0 --double_dqn --net_type 2 --data_dir ../data/networks2/ --log_file ../data/networks2/logs
elif [ $1 -eq 3 ]
then
    export CUDA_VISIBLE_DEVICES=3
    python tf_luncher.py --rom_name seaquest.bin --cut_top 0 --data_dir ../data/networks3/ --log_file ../data/networks3/logs
else
    echo 'wrong argument'
fi

#python tf_luncher.py --rom_name $1
