#!/bin/bash
#Submit to GPU
DEVICES=4
MODEL_PATH="/scratch_net/ofsoundof/yawli/logs_basis_sr/Test/model_github"

#################################################################
# Table 2
#################################################################
# EDSR-8-128: Baseline

CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR --save EDSR_X4_L8 --scale 4 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X4_L8.pt" --chop

# EDSR-8-128: Basis-128-27
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X4_L8_B128+27 --scale 4 --basis_size 128 --n_basis 27 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L8_B128+27.pt" --chop

# EDSR-8-128: Basis-128-40
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X4_L8_B128+40 --scale 4 --basis_size 128 --n_basis 40 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L8_B128+40.pt" --chop

# EDSR-8-128: Factor-SIC2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Factor --save EDSR_Factor_X4_L8_SIC2 --scale 4 --sic_layer 2 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L8_SIC2.pt" --chop

# EDSR-8-128: Factor-SIC3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Factor --save EDSR_Factor_X4_L8_SIC3 --scale 4 --sic_layer 3 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L8_SIC3.pt" --chop

# SRResNet: Baseline
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model SRResNet --save SRResNet_X4_L16 --scale 4 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_X4_L16.pt" --chop

# SRResNet: Basis-64-14
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model SRResNet_Basis --save SRResNet_Basis_X4_L16_B64+14 --scale 4 --basis_size 64 --n_basis 14 --bn_every --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Basis_X4_L16_B64+14.pt" --chop

# SRResNet: Basis-32-32
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model SRResNet_Basis --save SRResNet_Basis_X4_L16_B32+32 --scale 4 --basis_size 32 --n_basis 32 --bn_every --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Basis_X4_L16_B32+32.pt" --chop

# SRResNet: Factor-SIC2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model SRResNet_Factor --save SRResNet_Factor_X4_L16_SIC2 --scale 4 --sic_layer 2 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Factor_X4_L16_SIC2.pt" --chop

# SRResNet: Factor-SIC3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model SRResNet_Factor --save SRResNet_Factor_X4_L16_SIC3 --scale 4 --sic_layer 3 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Factor_X4_L16_SIC3.pt" --chop


#################################################################
# Table 3
#################################################################
# EDSR: Baseline X2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR --save EDSR_X2_32 --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X2_L32.pt" --chop

# EDSR: Baseline X3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR --save EDSR_X3_32 --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X3_L32.pt" --chop

# EDSR: Baseline X4
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR --save EDSR_X4_32 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X4_L32.pt" --chop

# EDSR: Basis X2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X2_L32 --scale 2 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X2_L32.pt" --chop

# EDSR: Basis X3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X3_L32 --scale 3 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X3_L32.pt" --chop

# EDSR: Basis X4
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X4_L32 --scale 4 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L32.pt" --chop

# EDSR: Basis-S X2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X2_L32U --scale 2 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X2_L32U.pt" --chop

# EDSR: Basis-S X3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X3_L32U --scale 3 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X3_L32U.pt" --chop

# EDSR: Basis-S X4
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Basis --save EDSR_Basis_X4_L32U --scale 4 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L32U.pt" --chop

# EDSR: Factor X2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Factor --save EDSR_Factor_X2_L32 --scale 2 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X2_L32.pt" --chop

# EDSR: Factor X3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Factor --save EDSR_Factor_X3_L32 --scale 3 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X3_L32.pt" --chop

# EDSR: Factor X4
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --model EDSR_Factor --save EDSR_Factor_X4_L32 --scale 4 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L32.pt" --chop



