#!/usr/bin/bash
set -e

# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}

echo '--specify the path of training dataset, style image and output ...'
training_dataset=/media/james/alibaba1/FILES_OFFICE/Researches/Style-transfer/Fast-neural-style-transfer/dataset_h5/coco-dataset.h5
style_image=./images/styles/newyork_night5.jpg
output=./models/pre-trained/newyork_night
echo '--run training process for model...'

th train.lua -h5_file ${training_dataset} -style_image ${style_image} -style_image_size 384 -pixel_loss_weight 0.0 -content_weights 1.0 -checkpoint_name ${output} -gpu 0 -content_layers 4,9,16

echo '--training process is done.'
echo '--the output model is located in ${output}'
