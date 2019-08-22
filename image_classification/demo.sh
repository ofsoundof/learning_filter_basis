#!bin/bash

DEVICES=6
MODEL_PATH="/scratch_net/ofsoundof/yawli/logs_basis_classification/model_classification"

# VGG_Factor
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template VGG --model VGG_Factor --vgg_type 16 --save Test/VGG_Factor --test_only --pretrained "${MODEL_PATH}/VGG_Factor.pt"

# VGG_Group
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template VGG --model VGG_Group --vgg_type 16 --save Test/VGG_Group_Gs64 --test_only --group_size 64 --pretrained "${MODEL_PATH}/VGG_Group.pt" 

# VGG_Basis
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template VGG --model VGG_Basis --vgg_type 16 --save Test/VGG_Basis --test_only --pretrained "${MODEL_PATH}/VGG_Basis.pt" --basis_size 128

# ResNet_Factor
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template ResNet --model ResNet_Factor --depth 56 --save Test/ResNet_Factor --test_only --pretrained "${MODEL_PATH}/ResNet_Factor.pt"

# ResNet_Group
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template ResNet --model ResNet_Group --depth 56 --group_size 8 --save Test/ResNet_Group --test_only --pretrained "${MODEL_PATH}/ResNet_Group.pt"

# ResNet_Basis
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template ResNet --model ResNet_Basis --depth 56 --k_size2 1 --basis_size 128 --save Test/ResNet_Basis --test_only --pretrained "${MODEL_PATH}/ResNet_Basis.pt"

# DenseNet_Factor
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template DenseNet --model DenseNet_Factor --depth 40 --save Test/DenseNet_Factor --test_only --pretrained "${MODEL_PATH}/DenseNet_Factor.pt"

# DenseNet_Group
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template DenseNet --model DenseNet_Group --group_size 6 --depth 40 --save Test/DenseNet_Group --test_only --pretrained "${MODEL_PATH}/DenseNet_Group.pt"

# DenseNet_Basis
CUDA_VISIBLE_DEVICES=$DEVICES python main.py --template DenseNet --model DenseNet_Basis --transition_group 6 --depth 40 --n_basis 36 --save Test/DenseNet_Basis --test_only --pretrained "${MODEL_PATH}/DenseNet_Basis.pt"








