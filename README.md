# Caffe-ExcitationBP-RNNs

This is a Caffe implementation of Excitation Backprop for RNNs described in

> [Sarah Adel Bargal*, Andrea Zunino*, Donghyun Kim, Jianming Zhang, Vittorio Murino, Stan Sclaroff. "Excitation Backprop for RNNs." CVPR, 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Bargal_Excitation_Backprop_for_CVPR_2018_paper.html)

__This software implementation is provided for academic research and non-commercial purposes only.  This implementation is provided without warranty.__

> [Repo for Excitation Backprop for CNNs](https://github.com/jimmie33/Caffe-ExcitationBP)

## Prerequisites
1. The same prerequisites as Caffe
2. Anaconda (python packages)

## Quick Start
1. Unzip the files to a local folder (denoted as **root_folder**).
2. Enter the **root_folder** and compile the code the same way as in [Caffe](http://caffe.berkeleyvision.org/installation.html).
  - Our code is tested in GPU mode, so make sure to activate the GPU code when compiling the code.
  - Make sure to compile pycaffe, the python interface
3. Enter **root_folder/excitationBP-RNNs**, run **demo.ipynb** using the python notebook. It will show you how to compute the spatiotemporal saliency maps of a video, and includes the examples in the demo video. For details of running the python notebook remotely on a server, see [here](https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh).

## Other comments
1. We implemented both GPU and CPU versions of Excitation Backprop for RNNs. Change `caffe.set_mode_eb_gpu()` to `caffe.set_mode_eb_cpu()` to run the CPU version.
2. You can download a pre-trained action recognition model at [this link](https://www.dropbox.com/sh/vxn6xkzujtnmody/AABqBVIGXGXbzFO3b5LE8hQWa?dl=0). The model must be placed in the folder **root_folder/models/VGG16_LSTM/**
3. To apply your own CNN-LSTM model, you need to modify **root_folder/models/VGG16_LSTM/deploy.prototxt**. You need to add a dummy loss layer at the end of the file.
