# Learning Filter Basis for Convolutional Neural Network Compression: Image Classification Task


## Quick Start (Test)
1. `git clone https://github.com/ofsoundof/learning_filter_basis.git`
2. CIFAR10 dataset will be automatically downloaded if it does not exist.
3. Download the test model from [Google drive](https://drive.google.com/file/d/1OQJ-JzSs3qhP79_dRJAYu6J7g1bN5rgq/view?usp=sharing).
4. Be sure to change the following directories in [`./option.py`](./option.py).
	
	`--dir_data`: the directory where you put CIFAR10 dataset.
	
	`--dir_save`: the directory where you want to save your the results.
5. Go to [`./demo.sh`](./demo.sh) and change `DEVICES` to your available CUDA devices and `MODEL_PATH` to the directory where you put the test model.
6. Run the demo commands in [`./demo.sh`](./demo.sh).


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



