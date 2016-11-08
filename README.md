# Tensorflow-SegNet
Implement slightly different [SegNet](http://arxiv.org/abs/1511.00561) in tensorflow,
successfully trained segnet-basic in CamVid dataset

for model detail, please go to https://github.com/alexgkendall/caffe-segnet
# Remark
Due to indice unravel still unavailable in tensorflow, the original upsampling
method is temporarily replaced by deconv( or conv-transpose) layer.
this model is still under construction, mainly use for personal research

# Dataset
The goal of this project is to train SegNet on publicly available dataset of faces: http://vis-www.cs.umass.edu/lfw/part_labels/ modified to contain only two categories (background, face + hair). Due to the large amount of data in the input dataset (original faces), only targets (black and white images) are saved in this repo. 
