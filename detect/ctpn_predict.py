#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
# print(curPath)# print(rootPath)
sys.path.append(rootPath)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from detect.ctpn_model import CTPN_Model
from detect.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from detect.ctpn_utils import resize
from detect import config

prob_thresh = 0.5
height = 720
gpu = True
if not torch.cuda.is_available():
    gpu = False
device = torch.device('cuda:0' if gpu else 'cpu')
weights = os.path.join(config.checkpoints_dir, 'CTPN_ep40.pth')
# weights = os.path.join('checkpoints\CTPN.pth')
# print(config.checkpoints_dir)
model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_det_boxes(image,display = True, expand = True):
    image = resize(image, height=height)
    image_r = image.copy()
    image_c = image.copy()
    h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        image = image.to(device)
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])
        # print(bbox.shape)

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        # print(np.max(cls_prob[0, :, 1]))
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        # print(select_anchor.shape)
        keep_index = filter_bbox(select_anchor, 16)

        # nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        # print(keep)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])

        # expand text
        if expand:
            for idx in range(len(text)):
                text[idx][0] = max(text[idx][0] - 10, 0)
                text[idx][2] = min(text[idx][2] + 10, w - 1)
                text[idx][4] = max(text[idx][4] - 10, 0)
                text[idx][6] = min(text[idx][6] + 10, w - 1)


        # print(text)
        if display:
            blank = np.zeros(image_c.shape,dtype=np.uint8)
            for box in select_anchor:
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1)
            image_c = image_c+blank
            image_c[image_c>255] = 255
            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2) #上边 左上角1开始 右上角3结束 (列,行)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2) #左边 
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2) #右边 
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2) #下边 左下角5开始 右下角7结束
                cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)
            # dis(image_c)
        # print(text)
        return text,image_c,image_r

if __name__ == '__main__':
    img_path = 'test_images\22.png'
    image = cv2.imread(img_path)
    text,image = get_det_boxes(image)
    dis(image)