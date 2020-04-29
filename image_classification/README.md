# Learning Filter Basis for Convolutional Neural Network Compression: Image Classification Task

## Train
1. CIFAR10 dataset will be automatically downloaded if it does not exist.
2. Be sure to change the following directories in [`./option.py`](./option.py).
	
	`--dir_data`: the directory where you put CIFAR10 dataset.
	
	`--dir_save`: the directory where you want to save your the results.
2. `cd ./scripts` and run the following scripts to train models.
    ```bash
    # VGG
	python ../main.py --template VGG --model VGG_Basis --vgg_type 16 --save VGG_Basis --basis_size 128

    # ResNet
    python ../main.py --template ResNet --model ResNet_Basis --batch_size 64 --epochs 400 --decay step-250-325 --depth 56 --k_size2 1 --save ResNet_Basis 
    
    # DenseNet
    python ../main.py --template DenseNet --model DenseNet_Basis --batch_size 64 --epochs 400 --decay step-250-325 --transition_group 6 --depth 40 --n_basis 36 --save DenseNet_Basis
    ```
    You can use scripts in `./scripts/train_XXX.sh` to reproduce the results in our paper. '`XXX`' denotes `vgg`, `densenet` and `resnet`.

## Test
1. Download the test model from [GoogleDrive](https://drive.google.com/file/d/1OQJ-JzSs3qhP79_dRJAYu6J7g1bN5rgq/view?usp=sharing) or [BaiduPan (Extraction Code: qesq)](https://pan.baidu.com/s/18MP02c_j0tHpdlqkf9mUBQ).
2. Go to [`./scripts/demo.sh`](./scripts/demo.sh) and change `DEVICES` to your available CUDA devices and `MODEL_PATH` to the directory where you put the test model.
3. Run the demo commands in [`./scripts/demo.sh`](./scripts/demo.sh).
    ```bash
    DEVICES=1
    MODEL_PATH="/home/thor/Downloads/model_classification"

    # VGG_Factor
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model VGG_Factor --vgg_type 16 --save Test/VGG_Factor --test_only --pretrained "${MODEL_PATH}/VGG_Factor.pt"

    # VGG_Group
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model VGG_Group --vgg_type 16 --save Test/VGG_Group_Gs64 --test_only --group_size 64 --pretrained "${MODEL_PATH}/VGG_Group.pt" 

    # VGG_Basis
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template VGG --model VGG_Basis --vgg_type 16 --save Test/VGG_Basis --test_only --pretrained "${MODEL_PATH}/VGG_Basis.pt" --basis_size 128 


    # ResNet_Factor
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ResNet_Factor --depth 56 --save Test/ResNet_Factor --test_only --pretrained "${MODEL_PATH}/ResNet_Factor.pt" 

    # ResNet_Group
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ResNet_Group --depth 56 --group_size 8 --save Test/ResNet_Group --test_only --pretrained "${MODEL_PATH}/ResNet_Group.pt" 

    # ResNet_Basis
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template ResNet --model ResNet_Basis --depth 56 --k_size2 1 --basis_size 128 --save Test/ResNet_Basis --test_only --pretrained "${MODEL_PATH}/ResNet_Basis.pt" 


    # DenseNet_Factor
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model DenseNet_Factor --depth 40 --save Test/DenseNet_Factor --test_only --pretrained "${MODEL_PATH}/DenseNet_Factor.pt" 

    # DenseNet_Group
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model DenseNet_Group --group_size 6 --depth 40 --save Test/DenseNet_Group --test_only --pretrained "${MODEL_PATH}/DenseNet_Group.pt" 

    # DenseNet_Basis
    CUDA_VISIBLE_DEVICES=$DEVICES python ../main.py --template DenseNet --model DenseNet_Basis --transition_group 6 --depth 40 --n_basis 36 --save Test/DenseNet_Basis --test_only --pretrained "${MODEL_PATH}/DenseNet_Basis.pt" 
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



