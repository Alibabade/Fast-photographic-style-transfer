# Fast photographic style transfer

This is the torch implementation code for the fast photographic style transfer based on Fast-Neural-Style(http://cs.stanford.edu/people/jcjohns/eccv16/)


Stylizing an image at a resolution of 852x480 approximately takes **40 milliseconds** on a single GTX 1080Ti Card:

<div align='center'>
  <img src='images/contents/hongkong_night.jpg' height="225px">
  <img src='images/styles/newyork_night.jpg' height="225px">
  <img src='results/hongkong-2-newyork.png' height="346px">
</div>

In this repo, we also provide the torch implementation of the Domain Transform (Recursive Filter) described in the paper:

    Domain Transform for Edge-Aware Image and Video Processing
    Eduardo S. L. Gastal and Manuel M. Oliveira
    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
    Proceedings of SIGGRAPH 2011, Article 69.


## Setup
All code is implemented in [Torch](http://torch.ch/).

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
```

### (Optional) GPU Acceleration

If you have an NVIDIA GPU, you can accelerate all operations with CUDA.

First [install CUDA](https://developer.nvidia.com/cuda-downloads), then
update / install the following packages:

```bash
luarocks install cutorch
luarocks install cunn
```

### (Optional) cuDNN

When using CUDA, you can use cuDNN to accelerate convolutions.

First [download cuDNN](https://developer.nvidia.com/cudnn) and copy the
libraries to `/usr/local/cuda/lib64/`. Then install the Torch bindings for cuDNN:

```bash
luarocks install cudnn
```

### Pretrained Models
The pre-trained models are located in './models/pre-trained/'


## Running examples

The script `fast_photographic_style_transfer_single.sh` lets you use a trained model to 
stylize example images:
```bash
bash fast_photographic_style_transfer_single.sh

```
### Running on new images
Change the image paths in `fast_photographic_style_transfer_single.sh` allow you to use a trained 
model to stylize new images:
```bash
th fast_neural_style.lua \
  -model ./models/pre-trained/newyorknight.t7 \
  -image_size 700 \
  -width 0 \
  -median_filter 3 \
  -timing 1 \
  -input_image [path to your content image] \
  -output_image [path to your temporal output image] \
  -gpu 0

th StyleFusion.lua \
  -Type single \
  -content_image [path to your content image] \
  -stylized_image [path to your temporal output image] \
  -output_image [path to your final output image]
```

### Running on videos
Change the video and frame paths in `fast_photographic_style_transfer_video.sh` allow you to use a trained 
model to stylize a new video:

```bash
ffmpeg -i [path to your video] [path to your frames]/frame_%04d.ppm

th fast_neural_style.lua \
  -model ./models/pre-trained/newyork_night.t7 \
  -image_size 700 \
  -width 0 \
  -median_filter 3 \
  -timing 1 \
  -input_dir [path to your frames]/frame_%04d.ppm \
  -output_dir [path to your tmp stylized frames] \
  -gpu 0

echo '--run Post-processing step...'
th StyleFusion.lua \
  -Type video \
  -input_pattern [path to your frames]/frame_%04d.ppm \
  -stylized_pattern [path to your tmp stylized frames]/frame_%04d.ppm \
  -output_pattern [path to your final frames]/frame_%04d.png


ffmpeg -i [path to your final frames]/frame_%04d.png [path to your video output]/stylized.mp4
```


### Some example results


<div align='center'>
  <img src='images/contents/2.jpg' height="185px">
  <img src='images/styles/luca-micheli.jpg' height="185px">
  <img src='results/single/2_final.jpg' height="185px">
  <img src='images/contents/5.jpg' height='185px'>
  <img src='images/styles/dawid-zawila.jpg' height='185px'>
  <img src='results/single/5_final.jpg' height="185px">
  <img src='images/contents/6.jpg' height='185px'>
  <img src='images/styles/street_sunset.jpg' height='185px'>
  <img src='results/single/6_final.jpg' height="185px">
  <img src='images/contents/7.jpg' height='185px'>
  <img src='images/styles/wade_meng.jpg' height='185px'>
  <img src='results/single/7_final.jpg' height="185px">
  <img src='images/contents/8.jpg' height='185px'>
  <img src='images/styles/tokyo-sunset.jpg' height='185px'>
  <img src='results/single/8_final.jpg' height="185px">
  <img src='images/contents/9.jpg' height='185px'>
  <img src='images/styles/girl-face2.jpg' height='185px'>
  <img src='results/single/9_final.jpg' height="185px">
</div>



## Parameters

You can control the size of the output images using the `-image_size` flag.

By default this script runs on CPU; to run on GPU, add the flag `-gpu`
specifying the GPU on which to run.

The full set of options for this script is [described here](doc/flags.md#fast_neural_stylelua).


## Training new models

You can [find instructions for training new models here](doc/training.md).

The training example is described in 'run_train.sh'.


The code builds on
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)
by Justin Johnson, Alexandre Alahi, and Li Fei-Fei.

## License

Free for personal or research use; for commercial use please contact me.
