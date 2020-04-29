#!/bin/bash
#Submit to GPU


DEVICES=0


###########################
# Basis: Learning Filter Basis for Convolutional Neural Network Compression. ICCV 2019
###########################
MODEL=ResNet_Basis
K=1
CHECKPOINT=${MODEL}_K${K}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ${MODEL} --batch_size 64 --epochs 400 --decay step-250-325 \
--depth 56 --k_size2 ${K} \
--save $CHECKPOINT 


###########################
# Factor: Factorized Convolutional Neural Networks. ICCV Workshop.
###########################
MODEL=ResNet_Factor
CHECKPOINT=${MODEL}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ${MODEL} --batch_size 64 --epochs 300 --decay step-150-225 \
--depth 56 \
--save $CHECKPOINT 


###########################
# Group: Extreme Network Compression via Filter Group Approximation. ECCV
###########################
MODEL=ResNet_Group
Gs=8
CHECKPOINT=${MODEL}_Gs${Gs}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ${MODEL} --batch_size 64 --epochs 300 --decay step-150-225 \
--depth 56 --group_size $Gs \
--save $CHECKPOINT 


