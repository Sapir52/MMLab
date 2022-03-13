# Commented out IPython magic to ensure Python compatibility.
# install dependencies: (use cu111 because colab has CUDA 11.1)
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdetection
!rm -rf mmdetection
!git clone https://github.com/open-mmlab/mmdetection.git
# %cd mmdetection

!pip install -e .

from mmcv import collect_env
collect_env()
# Check Pytorch installation
import torch, torchvision
# Check MMDetection installation
import mmdet
# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector
import sys

def demo_mmdet():
  # Choose to use a config and initialize the detector
  config_file = 'configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
  # Setup a checkpoint file to load
  checkpoint_file = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

  # build the model from a config file and a checkpoint file
  model = init_detector(config_file, checkpoint_file, device='cuda:0')

  # test a single image and show the results
  img = 'demo/demo.jpg' # or img = mmcv.imread(img), which will only load it once
  result = inference_detector(model, img)
  # visualize the results in a new window
  model.show_result(img, result,show=True)
  # or save the visualization results to image files
  model.show_result(img, result, out_file='demo/result.jpg')

def video_mmdet():
  # test a video and show the results
  video = mmcv.VideoReader('/content/noViolence.mp4')
  for frame in video:
      result = inference_detector(model, frame)
      model.show_result(frame, result,show=True)


"""#MMDet3D"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/open-mmlab/mmsegmentation.git
# %cd mmsegmentation
!pip install -r requirements.txt
!python setup.py develop

import mmseg

# Commented out IPython magic to ensure Python compatibility.
# Install mmdetection
!git clone https://github.com/open-mmlab/mmdetection3d.git
# %cd mmdetection3d
!pip install -r requirements.txt
!python setup.py develop

import os
import mmdet3d
from mmdet3d.apis import inference_detector, init_model#, show_result_meshlab

def demo_mmdet3d():
  
  config_file = 'configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'  
  checkpoint_file = 'hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth'

  # build the model from a config file and a checkpoint file
  model = init_model(config_file, checkpoint_file, device='cuda:0')

  # test a single bin
  pcd = 'demo/data/kitti/kitti_000008.bin'
  result, data = inference_detector(model, pcd)

  # show the results
  model.show_results(data, result, out_dir='/content/res')

if __name__ == '__main__':
  # We download the pre-trained checkpoints for inference and finetuning -for mmdet.
  !mkdir checkpoints
  !wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth \
        -O checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
  demo_mmdet()
  video_mmdet()
  # for mmdet3d
  !wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth
  demo_mmdet3d()
