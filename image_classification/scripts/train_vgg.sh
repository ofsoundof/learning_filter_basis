#!bin/bash


DEVICES=1


###########################
# Basis: Learning Filter Basis for Convolutional Neural Network Compression. ICCV 2019
###########################
MODEL=VGG_Basis
S_BASIS=128
CHECKPOINT=${MODEL}_Base${S_BASIS}
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model ${MODEL} --vgg_type 16 --save ${CHECKPOINT} --basis_size ${S_BASIS}


###########################
# Factor: Factorized Convolutional Neural Networks. ICCV Workshop.
###########################
MODEL=VGG_Factor
CHECKPOINT=${MODEL}
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model ${MODEL} --vgg_type 16 --save ${CHECKPOINT} 


###########################
# Group: Extreme Network Compression via Filter Group Approximation. ECCV
###########################
MODEL=VGG_Group
S_GROUP=128
CHECKPOINT=${MODEL}_Gs${S_GROUP}
CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model ${MODEL} --vgg_type 16 --save ${CHECKPOINT} --group_size ${S_GROUP} 

