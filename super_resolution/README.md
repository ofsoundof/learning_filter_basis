# Learning Filter Basis for Convolutional Neural Network Compression: Image Super-Resolution Task

## Dataset structure

    super_resolution
    ├── DIV2K
    │   ├── DIV2K_train_HR
    │   ├── DIV2K_train_LR_bicubic
    │   │   ├── X2
    │   │   ├── X3
    │   │   ├── X4
    │   ├── DIV2K_valid_HR
    │   ├── DIV2K_valid_LR_bicubic
    │   │   ├── X2
    │   │   ├── X3
    │   │   └── X4
    │   └──
    ├── benchmark
    │   ├── Set5
    │   ├── Set14
    │   ├── B100
    │   ├── Urban100
    │   └──
    └──

## Train
1. Prepare image super-resolution dataset.

   i.   Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) training and validation images. Please download the low-resolution images in the NTIRE-2017 challenge. 

   ii.  Download Set5, Set14, B100, and Urban100 benchmark from [GoogleDrive](https://drive.google.com/file/d/1y8kIpiAa5s-fZ_R5pd4Aq2wJFPfgXFxB/view?usp=sharing) or [BaiduPan (extraction code: e5ae)](https://pan.baidu.com/s/1al55wVZyuDkgKogY6TwtVw).
   
   iii. Put the images in a folder called `super_resolution`. So the folder structure should be like the one above.

2. `cd ./scripts` and run the following scripts to train the models.
    ```bash
    # EDSR-8-128: Basis-128-27
    CUDA_VISIBLE_DEVICES=1 python ../main.py --model EDSR_Basis --save EDSR_Basis --scale 4 --patch_size 192 --batch_size 16 --basis_size 128 \
    --n_basis 27 --n_resblocks 8 --n_feats 128 --res_scale 1 \
    --epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 --save_results

    # SRResNet: Basis-32-32
    CUDA_VISIBLE_DEVICES=1 python ../main.py --model SRResNet_Basis --save SRResNet_Basis --scale 4 --patch_size 96 --batch_size 16 \
    --basis_size 32 --n_basis 32 --n_resblocks 16 --n_feats 64 --bn_every \
    --epochs 300 --lr_decay 200 --lr 0.0001 --n_threads 8 --n_train 32208 --data_train DIV2KSUB --data_test Set5 --save_results
    ```
    
    For more information, please refer to the train scripts in `./scripts`(./scripts)

## Test
1. Download the test model from [GoogleDrive](https://drive.google.com/file/d/1dUi2GVO2QD6kNwYY71ZOA1vw0BxwNyfU/view?usp=sharing) or [BaiduPan (Extraction Code: vvfr)](https://pan.baidu.com/s/17ectGT1UkE-hsR2hhOJVBw).
2. Be sure to change the following directories in [`./option.py`](./option.py).

	`--dir_data`: the directory where you put the dataset.

	`--dir_save`: the directory where you want to save your the results.
3. Go to [`./scripts/demo.sh`](./scripts/demo.sh) and change `DEVICES` to your available CUDA devices and `MODEL_PATH` to the directory where you put the test model.
4. Run the demo commands in [`./scripts/demo.sh`](./scripts/demo.sh).
    ```bash
    DEVICES=1
    MODEL_PATH="/home/thor/Downloads/model_super_resolution"

    #################################################################
    # Table 2
    #################################################################
    # EDSR-8-128: Baseline

    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR --save EDSR_X4_L8 --scale 4 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X4_L8.pt" --chop

    # EDSR-8-128: Basis-128-27
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X4_L8_B128+27 --scale 4 --basis_size 128 --n_basis 27 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L8_B128+27.pt" --chop

    # EDSR-8-128: Basis-128-40
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X4_L8_B128+40 --scale 4 --basis_size 128 --n_basis 40 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L8_B128+40.pt" --chop

    # EDSR-8-128: Factor-SIC2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Factor --save EDSR_Factor_X4_L8_SIC2 --scale 4 --sic_layer 2 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L8_SIC2.pt" --chop

    # EDSR-8-128: Factor-SIC3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Factor --save EDSR_Factor_X4_L8_SIC3 --scale 4 --sic_layer 3 --n_resblocks 8 --n_feats 128 --res_scale 1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L8_SIC3.pt" --chop

    # SRResNet: Baseline
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model SRResNet --save SRResNet_X4_L16 --scale 4 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_X4_L16.pt" --chop

    # SRResNet: Basis-64-14
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model SRResNet_Basis --save SRResNet_Basis_X4_L16_B64+14 --scale 4 --basis_size 64 --n_basis 14 --bn_every --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Basis_X4_L16_B64+14.pt" --chop

    # SRResNet: Basis-32-32
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model SRResNet_Basis --save SRResNet_Basis_X4_L16_B32+32 --scale 4 --basis_size 32 --n_basis 32 --bn_every --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Basis_X4_L16_B32+32.pt" --chop

    # SRResNet: Factor-SIC2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model SRResNet_Factor --save SRResNet_Factor_X4_L16_SIC2 --scale 4 --sic_layer 2 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Factor_X4_L16_SIC2.pt" --chop

    # SRResNet: Factor-SIC3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model SRResNet_Factor --save SRResNet_Factor_X4_L16_SIC3 --scale 4 --sic_layer 3 --n_resblocks 16 --n_feats 64 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/SRResNet_Factor_X4_L16_SIC3.pt" --chop


    #################################################################
    # Table 3
    #################################################################
    # EDSR: Baseline X2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR --save EDSR_X2_32 --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X2_L32.pt" --chop

    # EDSR: Baseline X3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR --save EDSR_X3_32 --scale 3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X3_L32.pt" --chop

    # EDSR: Baseline X4
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR --save EDSR_X4_32 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_X4_L32.pt" --chop

    # EDSR: Basis X2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X2_L32 --scale 2 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X2_L32.pt" --chop

    # EDSR: Basis X3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X3_L32 --scale 3 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X3_L32.pt" --chop

    # EDSR: Basis X4
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X4_L32 --scale 4 --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L32.pt" --chop

    # EDSR: Basis-S X2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X2_L32U --scale 2 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X2_L32U.pt" --chop

    # EDSR: Basis-S X3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X3_L32U --scale 3 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X3_L32U.pt" --chop

    # EDSR: Basis-S X4
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Basis --save EDSR_Basis_X4_L32U --scale 4 --share_basis --basis_size 256 --n_basis 32 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Basis_X4_L32U.pt" --chop

    # EDSR: Factor X2
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Factor --save EDSR_Factor_X2_L32 --scale 2 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X2_L32.pt" --chop

    # EDSR: Factor X3
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Factor --save EDSR_Factor_X3_L32 --scale 3 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X3_L32.pt" --chop

    # EDSR: Factor X4
    CUDA_VISIBLE_DEVICES=${DEVICES} python ../main.py --model EDSR_Factor --save EDSR_Factor_X4_L32 --scale 4 --sic_layer 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set5 --save_results --n_GPUs 1 --test_only --pre_train "${MODEL_PATH}/EDSR_Factor_X4_L32.pt" --chop
    ```

## Reference
If you find our work useful in your research of publication, please cite our work:

```
@inproceedings{li2019learning,
  title = {Learning Filter Basis for Convolutional Neural Network Compression},
  author = {Li, Yawei and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year = {2019}
}
```



