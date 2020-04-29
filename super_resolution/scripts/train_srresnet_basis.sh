#!/bin/bash
#Submit to GPU


# SRResNet: Basis-64-14
MODEL=SRResNet_Basis
N_BLOCK=16
N_FEATS=64
N_PATCH=96
N_BATCH=16
N_BASIS=14
S_BASIS=64
SCALE=4
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_BaseS${S_BASIS}N${N_BASIS}_P${N_PATCH}B${N_BATCH}"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --batch_size $N_BATCH \
--basis_size ${S_BASIS} --n_basis ${N_BASIS} --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --bn_every \
--epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 \
--save_results --print_model --n_GPUs 1  --reset


# SRResNet: Basis-32-32
MODEL=SRResNet_Basis
N_BLOCK=16
N_FEATS=64
N_PATCH=96
N_BATCH=16
N_BASIS=32
S_BASIS=32
SCALE=4
CHECKPOINT="${MODEL}_X${SCALE}_L${N_BLOCK}F${N_FEATS}_BaseS${S_BASIS}N${N_BASIS}_P${N_PATCH}B${N_BATCH}"
echo $CHECKPOINT

CUDA_VISIBLE_DEVICES=1 python ../main.py --model $MODEL --save $CHECKPOINT --scale $SCALE --patch_size $N_PATCH --batch_size $N_BATCH \
--basis_size ${S_BASIS} --n_basis ${N_BASIS} --n_resblocks $N_BLOCK --n_feats ${N_FEATS} --bn_every \
--epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 \
--save_results --print_model --n_GPUs 1  --reset
