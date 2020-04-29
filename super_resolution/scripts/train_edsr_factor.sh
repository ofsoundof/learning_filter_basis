#!/bin/bash
#Submit to GPU


# EDSR-8-128: Factor-SIC2
MODEL=EDSR_Factor
N_BLOCK=8
N_FEATS=128
N_PATCH=192
N_BATCH=16
SIC=2
SCALE=4
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}_P${N_PATCH}B${N_BATCH}"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --batch_size $N_BATCH \
--n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 --sic_layer ${SIC} \
--epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 \
--save_results --print_model --n_GPUs 1 --reset

# EDSR-8-128: Factor-SIC3
MODEL=EDSR_Factor
N_BLOCK=8
N_FEATS=128
N_PATCH=192
N_BATCH=16
SIC=3
SCALE=4
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}SIC${SIC}_P${N_PATCH}B${N_BATCH}"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --batch_size $N_BATCH \
--n_resblocks $N_BLOCK --n_feats ${N_FEATS} --res_scale 1 --sic_layer ${SIC} \
--epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 \
--save_results --print_model --n_GPUs 1 --reset

