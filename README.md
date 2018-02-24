# Cascaded Refinement Network


PyTorch implementation of paper: Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks.

Goal is to synthesise photographic images from semantic maps. Process can be seen as inverse segmentation.

<img src="https://s17.postimg.org/cnjnyds8f/image.png" width="600"><img src="https://s17.postimg.org/h9fs6qt73/image.png" width="600">

Cascaded Refinement Network is composed of refinement modules. Each module operates at different spatial scale. It takes as input concatenation of downsampled semantic map where each class gets different depth channel and output from previous module. 


<img src="https://s17.postimg.org/nadh4dden/crn.png" width="800">

Single refinement module Mi does basic operations like convolution, layer normalization and leaky ReLU.

<img src="https://s17.postimg.org/wux3r9i67/image.png" width="300">

The Cascaded Refinement Network generates not one, buk k diversed images. After passing each one of them through VGG19 network,  perceptual loss with respect to reference texture image from dataset is computed. Final loss is defined as minimum from perceptual losses computed for k diversed images.

<img src="https://s17.postimg.org/g8fjhowsv/loss.png" width="700">



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
