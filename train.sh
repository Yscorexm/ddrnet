#!/usr/bin/env bash

#expm=`basename $1`
expm=$1
logdir=`realpath log/$expm`
echo "logdir: "$logdir
time=`date +"%m-%d-%H-%M"`
logfile=${logdir}/log_$time.txt
mkdir -p $logdir
touch $logfile

cd src
CUDA_VISIBLE_DEVICES=0 python train.py \
    --index_file=../dataset/train.csv \
    --rand_crop \
    --rand_flip \
    --rand_depth_shift \
    --aux_type='PNG' \
    --diff_thres=2.5 \
    --low_thres=500 \
    --up_thres=3000 \
    --dnnet=convResnet \
    --dtnet=hypercolumn \
    --checkpoint_basename=$expm \
    --logdir=$logdir \
    --image_size=256 \
    --batch_size=16 \
    --learning_rate=0.001 \
    --max_steps=6000 \
    --save_model_steps=200 \
