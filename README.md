# Cascaded Refinement Networks


PyTorch implementation of paper: Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks.

Goal is to synthesise photographic images from semantic maps. 

<img src="https://s17.postimg.org/cnjnyds8f/image.png" width="600"><img src="https://s17.postimg.org/h9fs6qt73/image.png" width="600">


## Usage

1. Download VGG19 [model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat).

2. Download semantic maps and photographic images from [Cityscapes Dataset](https://www.cityscapes-dataset.com/), and put folders gtFine and leftImg8bit in  cityscape directory.

```
python semantic2labels.py
python train.py
python eval.py
```

<img src="https://s17.postimg.org/bl9hg0h6n/image.png" width="600"><img src="https://s17.postimg.org/57kecrzfz/36a.png" width="600">

<img src="https://s17.postimg.org/xvxcg748v/image.png" width="600"><img src="https://s17.postimg.org/68kn23qrz/53a.png" width="600">
