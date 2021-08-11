from __future__ import absolute_import


import numpy as np
import matplotlib
# matplotlib.use('Agg') #Agg 渲染器是非交互式的后端，没有GUI界面，所以不显示图片，但仍可以生成图像文件
matplotlib.use('TkAgg')
# matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from . import to_torch, to_numpy

def stn_vis(raw_images, rectified_images, ctrl_points):
  """
    raw_images: images without rectification
    rectified_images: rectified images with stn
    ctrl_points: predicted ctrl points
    preds: predicted label sequences
    targets: target label sequences
    real_scores: scores of recognition model
    pred_scores: predicted scores by the score branch
    dataset: xxx
    vis_dir: xxx
  """
  if raw_images.ndimension() == 3:
    raw_images = raw_images.unsqueeze(0)
    # rectified_images = rectified_images.unsqueeze(0)
  batch_size, _, raw_height, raw_width = raw_images.size()
  if batch_size>500:
        batch_size=500
  # translate the coordinates of ctrlpoints to image size
  ctrl_points = to_numpy(ctrl_points)
  ctrl_points[:,:,0] = ctrl_points[:,:,0] * (raw_width-1)
  ctrl_points[:,:,1] = ctrl_points[:,:,1] * (raw_height-1)
  ctrl_points = ctrl_points.astype(np.int)

  # tensors to pil images
  raw_images = raw_images.permute(0,2,3,1)
  with torch.no_grad():
    raw_images = to_numpy(raw_images)
  raw_images = (raw_images * 0.5 + 0.5)*255
  rectified_images = rectified_images.permute(0,2,3,1)
  rectified_images = to_numpy(rectified_images)
  rectified_images = (rectified_images * 0.5 + 0.5)*255

  # draw images on canvas
  vis_images = []
  num_sub_plot = 2
  raw_images = raw_images.astype(np.uint8)
  rectified_images = rectified_images.astype(np.uint8)

  # plt.ion() #修改处
  for i in range(batch_size):
      fig = plt.figure()
      ax = [fig.add_subplot(num_sub_plot,1,i+1) for i in range(num_sub_plot)]
      for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
      ax[0].imshow(raw_images[i])
      ax[0].scatter(ctrl_points[i,:,0], ctrl_points[i,:,1], marker='+', s=5)
      ax[1].imshow(rectified_images[i])
      # plt.subplots_adjust(wspace=0, hspace=0)
      plt.show() #若使用plt.ion则注释掉plt.show
      # mngr = plt.get_current_fig_manager()
      # mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置

      # plt.pause(0.1) #修改处
      plt.close()
    # plt.ioff() #修改处