import cv2
from math import *
import numpy as np
from detect.ctpn_predict import get_det_boxes
from recognize.crnn_recognizer import PytorchOcr
recognizer = PytorchOcr()
#------------------------------------#
from stn.model_builder import ModelBuilder
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torchvision import transforms
# from PIL import Image
from stn.stn_config import get_args
from utils.visualization_utils import stn_vis
from utils import to_torch, to_numpy
#--------------------------------------------------------#
args = get_args(sys.argv[1:])

def image_process(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    # img = Image.open(image_path).convert('RGB')
    if keep_ratio:
        # w, h = img.size
        h,w = img.shape[:2]
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    # img = img.resize((imgW, imgH), Image.BILINEAR)
    dim = (imgW, imgH)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
#--------------------------------------------------------#
def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def sort_box(box):
    """
    对box进行排序 4个顶点行坐标
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut

def charRec(img, text_recs, args, adjust=False):
    #------------------------------------#
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    args.cuda = args.cuda and torch.cuda.is_available()
    model = ModelBuilder(STN_ON=args.STN_ON)
    
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model.eval()
    """
    加载OCR模型，进行字符识别
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]

    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4) # H,W
        # dis(partImg)
        h,w=partImg.shape[:2]
        if h < 1 or w < 1 or h > w:  # 过滤异常图片
            continue

        #-----------------------------------------------------------#
        # x = image_process(partImg, imgH=32,imgW=128) #imgW=int(w/h*32),keep_ratio=True
        # with torch.no_grad():
        #     x = x.to(device)
        # input_dict = {}
        # input_dict['images'] = x.unsqueeze(0)#[1,3,32,100]

        # with torch.no_grad():
        #     rectified_x,ctrl_points= model(input_dict)
        # # torch.cat(images)
        # stn_vis(x, rectified_x, ctrl_points)
        # rectified_x = rectified_x.squeeze(0)#降维
        # rectified_x = rectified_x.permute(1,2,0)
        # rectified_x = to_numpy(rectified_x)
        # rectified_x = normalization(rectified_x)*255
        # rectified_x = rectified_x.astype(np.uint8)
        # text = recognizer.recognize(rectified_x)
        #-----------------------------------------------------------#
        
        text = recognizer.recognize(partImg) #partImg  rectified_x
        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)  # 识别文字

    return results

def ocr(image):
    # global_args = get_args(sys.argv[1:])
    # detect
    text_recs, img_framed, image = get_det_boxes(image)
    text_recs = sort_box(text_recs)
    result = charRec(image, text_recs,args)
    return result, img_framed,text_recs
