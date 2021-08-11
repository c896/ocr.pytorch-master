import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import cv2

def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    #-----------------------------灰度图----------------------------------#
    # image_1 = np.array(Image.open(image_file).convert('L'))
    # # image= np.expand_dims(image_1,axis=-1) # image = image_1[:,np.newaxis] #法1
    # # h,w=list(image_1.shape) # image=np.zeros((h,w,3)) #image[:,:,0:2]=image_1 #法2
    # image = np.expand_dims(image_1,axis=2).repeat(3,axis=2) #法3
    #---------------------------------------------------------------------#
    result, image_framed,text_recs= ocr(image)
    k=0
    for i in text_recs:
        try :
            s = result[k][1]
        except :
            continue
        k += 1
        if len(s)>8:
            continue
        i = [int(j) for j in i]
        cv2.putText(image_framed, s, (i[0]+int((i[2]-i[0])/2), i[1]+13),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5,
                    (0,0,255),2,cv2.LINE_AA)
    
    plt.figure() #figsize=(4, 4)
    # plt.ion()  # 打开交互模式
    plt.axis('off')  # 不需要坐标轴
    plt.imshow(image_framed)
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置
    plt.pause(1)  # 该句显示图片15秒
    # plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题
    # plt.clf()  # 清空图片
    # plt.close()  # 清空窗口

    return result,image_framed


if __name__ == '__main__':
    image_files = glob('test_images/frames_right_left/*.*')
    result_dir = 'test_result'
    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt')
        
        print(txt_file)
        txt_f = open(txt_file, 'w')
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            txt_f.write(result[key][1]+'\n')
        txt_f.close()
        plt.clf()  # 清空图片
        plt.close()  # 清空窗口