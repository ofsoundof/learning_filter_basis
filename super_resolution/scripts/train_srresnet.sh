#!/bin/bash
#Submit to GPU


# SRResNet: Baseline
MODEL=SRResNet
N_BLOCK=16
N_FEATS=64
N_PATCH=96
N_BATCH=16
SCALE=4
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_P${N_PATCH}B${N_BATCH}"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --batch_size $N_BATCH \
--n_resblocks $N_BLOCK --n_feats ${N_FEATS} \
--epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 \
--save_results --print_model --n_GPUs 1  --reset
