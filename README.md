# Cascaded Refinement Networks



## Usage

1. Download VGG19 [model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat).

2. Download semantic maps and photographic images from [Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put folders gtFine and leftImg8bit in  cityscape directory.

```
python semantic2labels.py
python train.py
python eval.py
```
