import os
import xml.etree.ElementTree as ET

dirpath = r'xml_label/xml'  # 原来存放xml文件的目录
newdir = r'xml_label/labels/'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

dict_info = {'smoke': 0,'0':2}  # 有几个 属性 填写几个

for fp in os.listdir(dirpath):
    if fp.endswith('.xml'):
        with open(os.path.join(newdir, fp.split('.xml')[0] + '.txt'), 'w+') as f:
            f.write('')

        root = ET.parse(os.path.join(dirpath, fp)).getroot()
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')
        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text

        n=len(root.findall('object'))
        i = 0
        for child in root.findall('object'):  # 找到图片中的所有框
            i += 1
            sub = child.find('bndbox')  # 找到框的标注值并进行读取
            label = child.find('name').text
            # label_ = dict_info.get(label)
            # if label:
            #     label = label_
            # else:
            #     label = 0
            xmin = float(sub[0].text)
            ymin = float(sub[1].text)
            xmax = float(sub[2].text)
            ymax = float(sub[3].text)
            # w = (xmax - xmin)
            # h = (ymax - ymin)
            # try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            #     x_center = (xmin + xmax) / (2 * width)
            #     x_center = '%.6f' % x_center
            #     y_center = (ymin + ymax) / (2 * height)
            #     y_center = '%.6f' % y_center
            #     w = (xmax - xmin) / width
            #     w = '%.6f' % w
            #     h = (ymax - ymin) / height
            #     h = '%.6f' % h
            # except ZeroDivisionError:
            #     print(filename, '的 width有问题')
            # with open(os.path.join(newdir, fp.split('.xml')[0] + '.txt'), 'a+') as f:
                # f.write(' '.join([str(x_center), str(y_center), str(w), str(h), str(label_) + '\n']))
            f1 = open(os.path.join(newdir, fp.split('.xml')[0] + '.txt'), 'a+')
            if i < n:
                f1.write(' '.join([str(xmin), str(ymin), str(xmax), str(ymax), str(label) + '\n']))
            else:
                f1.write(' '.join([str(xmin), str(ymin), str(xmax), str(ymax), str(label)]))
        f1.close()
print('ok')