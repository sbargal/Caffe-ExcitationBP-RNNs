import numpy as np
import sys
import glob
import os
import h5py
import argparse
from skimage import transform, filters


caffe_root = '../'
sys.path.insert(0, '../python/')
import caffe

  
def image_processor(transformer, input_im, image_dim=224):
  resize = (240, 320)
  data_in = caffe.io.load_image(input_im)
  if not ((data_in.shape[0] == resize[0]) & (data_in.shape[1] == resize[1])):
    data_in = caffe.io.resize_image(data_in, resize)
  
  shift_x = (resize[0] - image_dim)/2
  shift_y = (resize[1] - image_dim)/2
  shift_data_in = data_in[shift_x:shift_x+image_dim,shift_y:shift_y+image_dim,:] 
  processed_image = transformer.preprocess('data',shift_data_in)
  return processed_image

def create_transformer(net, image_dim, flow):
  shape = (128,3,image_dim,image_dim)
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_raw_scale('data', 255)
  image_mean = [103.939, 116.779, 128.68]
  if flow:
    image_mean = [128, 128, 128]
  channel_mean = np.zeros((3,image_dim,image_dim))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  return transformer

def get_batches(net,frames):
  
  image_dim = 224
  
  
  
  flow=False   #FLAG
  
  frames_per_clip = len(frames) # number of frames per clip
  num_clip = 1  # number of clips per batch
  batch_size = num_clip * frames_per_clip # number of frames per batch
  
  transformer = create_transformer(net, image_dim, flow) 
  
  #frames = sorted(glob.glob('%s/*jpg' %vid_fpath))
  
  data = []
  clip = []
  
  
  cnt = 0
  for frame in frames:
    data.append(image_processor(transformer, frame))
    if cnt % frames_per_clip == 0:
      clip.append(0)
    else:
      clip.append(1)
    cnt += 1
    f=frame
 
 
  clip = np.array(clip)
  clip.resize(len(clip),1,1,1)
  data = np.array(data)

  data_batches=data
  clip_batches=clip
	
  return data_batches, clip_batches
