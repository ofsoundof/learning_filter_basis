# Learning Filter Basis for Convolutional Neural Network Compression
This repository is an official PyTorch implementation of the Paper **Learning Filter Basis for Convolutional Neural Network Compression** accepted by ICCV 2019. The training scripts are updated.

## Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Train and Test](#train-and-test)
4. [Results](#results)
5. [Reference](#reference)
6. [Acknowledgements](#acknowledgements)

## Introduction
Convolutional neural networks (CNNs) based solutions have achieved state-of-the-art performances for many computer vision tasks, including classification and super-resolution of images. Usually the success of these methods comes with a cost of millions of parameters due to stacking deep convolutional layers. Moreover, quite a large number of filters are also used for a single convolutional layer, which exaggerates the parameter burden of current methods. Thus, in this paper, we try to reduce the number of parameters of CNNs by learning a basis of the filters in convolutional layers. For the forward pass, the learned basis is used to approximate the original filters and then used as parameters for the convolutional layers. We validate our proposed solution for multiple CNN architectures on image classification and image super-resolution benchmarks and compare favorably to the existing state-of-the-art in terms of reduction of parameters and preservation of accuracy.

<img src="/figs/filter_reduction.png" width="400">

Comparison between different filter decomposition methods.

<img src="/figs/convert_to_convolution.png" width="400">

Illustration of the proposed basis learning method. Operations are converted to convolutions. Unlike the normal convolution, our method splits both the input feature map and the 3D filter along the channel dimension. A set of basis is learned for the ensemble of splits. Every split of the input feature map is convolved with the basis. A final 1x1 convolution generates the output.

## Dependencies
* Python 3.7.3
* PyTorch >= 1.1.0
* numpy
* matplotlib
* tqdm
* scikit-image
* easydict

## Train and Test

### Image classification
For image classfication, go to [`./image_classification`](./image_classification)

### Image super-resolution

For image super-resolution, go to [`./super_resolution`](./super_resolution)

## Results

### Image classification

![Classification](/figs/Table5.png)

### Image super-resolution

![SRResNet_EDSRlight](/figs/Table2.png)

![EDSR](/figs/Table3.png)


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

## Acknowledgements
This repository is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for making their EDSR codes public.




