#!bin/bash


DEVICES=1


###########################
# Basis: Learning Filter Basis for Convolutional Neural Network Compression. ICCV 2019
###########################
MODEL=DenseNet_Basis
N_BASIS=36
T=6
CHECKPOINT=${MODEL}_N${N_BASIS}T${T}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model ${MODEL} --batch_size 64 --epochs 400 --decay step-250-325 \
--transition_group ${T} --depth 40 --n_basis ${N_BASIS} --save ${CHECKPOINT}


###########################
# Factor: Factorized Convolutional Neural Networks. ICCV Workshop.
###########################
MODEL=DenseNet_Factor
CHECKPOINT=${MODEL}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model ${MODEL} --batch_size 64 --epochs 300 --decay step-150-225 \
--depth 40 --save $CHECKPOINT


###########################
# Group: Extreme Network Compression via Filter Group Approximation. ECCV
###########################
MODEL=DenseNet_Group
S_GROUP=6
CHECKPOINT=${MODEL}_Gs${S_GROUP}
echo $CHECKPOINT
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model ${MODEL} --batch_size 64 --group_size ${S_GROUP} --epochs 300 --decay step-150-225 \
--depth 40 --save $CHECKPOINT
