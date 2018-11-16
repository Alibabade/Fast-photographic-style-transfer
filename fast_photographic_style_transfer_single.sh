#!/usr/bin/bash
set -e 

echo '--run fast neural style transfer for temporal stylized results...'
th fast_neural_style.lua \
  -model ./models/pre-trained/1.t7 \
  -image_size 700 \
  -width 0 \
  -median_filter 3 \
  -timing 1 \
  -input_image ./images/contents/1.jpg \
  -output_image ./results/single/tmp_results/1_tmp.png \
  -gpu 0

echo '--run Post-processing step...'
th StyleFusion.lua \
  -Type single \
  -content_image ./images/contents/1.jpg \
  -stylized_image ./results/single/tmp_results/1_tmp.png \
  -output_image ./results/single/final_results/1_final.png

echo '--Photographic Style Transfer is done.'


#1-hoach-le-dinh
#2-luca-micheli
#3-city-sunset
#4-newyork_night
#5-dawid-zawila
#6-street_sunset
#7-wade_meng
#8-tokyo-sunset
#9-girl-face
#10-bridge




