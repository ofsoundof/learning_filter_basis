# Learning Filter Basis for Convolutional Neural Network Compression: Image Super-Resolution Task


## Quick Start (Test)
1. `git clone https://github.com/ofsoundof/learning_filter_basis.git`
2. Download image super-resolution dataset.
   i)   Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) training and validation images. Please download the low-resolution images in the NTIRE-2017 challenge. 
   ii)  Download Set5, Set14, B100, and Urban100 [benchmark](https://drive.google.com/file/d/1y8kIpiAa5s-fZ_R5pd4Aq2wJFPfgXFxB/view?usp=sharing).
   iii) Put the images in a folder called `super_resolution`. So the folder structure should be:
    
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

	
3. Download the test model from [Google drive](https://drive.google.com/file/d/1dUi2GVO2QD6kNwYY71ZOA1vw0BxwNyfU/view?usp=sharing).
4. Be sure to change the following directories in [./option.py](./option.py).
	`--dir_data`: the directory where you put the dataset.
	`--dir_save`: the directory where you want to save your the results.
5. Go to [./demo.sh](./demo.sh) and change `DEVICES` to your available CUDA devices and `MODEL_PATH` to the directory where you put the test model.
6. Run the demo commands in [./demo.sh](./demo.sh).


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



