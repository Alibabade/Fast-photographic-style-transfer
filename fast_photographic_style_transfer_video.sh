#!/usr/bin/bash
set -e 

echo '--extract the frames from video...'
ffmpeg -i ./images/video/hkn.mp4 ./images/video/HKNight_clips/frame_%04d.ppm

echo '--run fast neural style transfer for temporal stylized results...'
th fast_neural_style.lua \
  -model ./models/pre-trained/newyork_night.t7 \
  -image_size 700 \
  -width 0 \
  -median_filter 3 \
  -timing 1 \
  -input_dir ./images/video/HKNight_clips/ \
  -output_dir ./results/video/tmp/ \
  -gpu 0

echo '--run Post-processing step...'
th StyleFusion.lua \
  -Type video \
  -input_pattern ./images/video/HKNight_clips/frame_%04d.ppm \
  -stylized_pattern ./results/video/tmp/frame_%04d.ppm \
  -output_pattern ./results/video/final/frame_%04d.png

echo '--Photographic Style Transfer is done.'

ffmpeg -i ./results/video/final/frame_%04d.png ./results/video/stylized.mp4

