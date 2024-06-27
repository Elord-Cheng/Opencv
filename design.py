import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import statistics

def cut(img):
    # 裁剪图片，保留所要检测的位置
    img = cv2.resize(img, (800, 800))
    cut = int(img.shape[0] / 2)
    img = img[cut:700, :]
    return img

def rotate(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将图像进行二值化
    _, img_t = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
    # 找轮廓
    contours, hierarchy = cv2.findContours(img_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # 获取点集的最小外接矩形,rect包括矩形的中心点坐标,高度宽度及倾斜角度等
    rect = cv2.minAreaRect(contours[0])
    # 获取倾斜角度
    angle = rect[2]
    length_1, length_2 = int(rect[1][0]), int(rect[1][1])
    if length_1 <= length_2:
        angle = - (90 - angle)
    # 旋转产品
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rotate = cv2.warpAffine(img, M, (cols, rows))

    return img_rotate, angle

def cal_contours_box(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将图像进行二值化
    _, img_t = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    # 找轮廓
    contours, hierarchy = cv2.findContours(img_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 对轮廓进行可视化
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # 获取点集的最小外接矩形,rect包括矩形的中心点坐标,高度宽度及倾斜角度等
    rect = cv2.minAreaRect(contours[0])
    # 获取最小外接矩形的四个顶点坐标
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    list(box)
    list(rect)
    return box, rect

def cal_box(box):
    return np.min(box[:, 0]), np.max(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 1])

def Iou(box_1, box_2):
    box_1 = cal_box(box_1)
    box_2 = cal_box(box_2)
    # 计算交集的坐标范围
    x1_i = max(box_1[0], box_2[0])
    y1_i = max(box_1[2], box_2[2])
    x2_i = min(box_1[1], box_2[1])
    y2_i = min(box_1[3], box_2[3])
    # 计算交集的面积
    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    # 计算并集的面积
    box1_area = (box_1[1] - box_1[0] + 1) * (box_1[3] - box_1[2] + 1)
    box2_area = (box_2[1] - box_2[0] + 1) * (box_2[3] - box_2[2] + 1)
    union_area = box1_area + box2_area - intersection_area
    # 计算 IoU
    iou = intersection_area / union_area
    return iou

def find_img_feature(img):
    # 将图片提取出来，无关内容剔除
    box, _ = cal_contours_box(img)

    # # 画出外轮廓
    # img_rec = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
    # cv2.imshow('img_rec', img_rec)
    # cv2.waitKey(0)

    # 取出目标图像
    x_min_1, x_max_1, y_min_1, y_max_1 = cal_box(box)
    img = img[y_min_1:y_max_1, x_min_1:x_max_1]

    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_t = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_t', img_t)
    # cv2.waitKey(0)
    # 找轮廓
    kernel = np.ones((3, 3))
    img_erode = cv2.erode(img, kernel, iterations=1)
    img_canny = cv2.Canny(img_erode, 70, 120)

    contours, hierarchy1 = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(img_t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return hist, len(contours)

def detect(hist, length, max_contours):
    detect_result = []
    for i in range(len(hist)):
        # 把ok的分出
        if hist[i] < 25000:
            detect_result.append(0)
        # 把气泡的分出
        if hist[i] > 25000 and length[i] > max_contours:
            detect_result.append(1)
        # 把胶带的分出
        if hist[i] > 25000 and length[i] < max_contours:
            detect_result.append(2)

    return detect_result

def progress(img):
    # 裁剪图像
    img_original = cut(img)
    # 旋转图像至水平
    img, angle = rotate(img_original)

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_original)
    # plt.subplot(1, 2, 2)
    # plt.imshow(img)
    # plt.show()

    # 寻找外轮廓
    box, _ = cal_contours_box(img)
    # 计算IOU
    iou = Iou(img_box, box)
    # 缺陷检测
    # max: 2.4 min : 4.3 4.0
    hist, length = find_img_feature(img)

    return hist, length, angle, iou

def draw_confusion_matrix(result):
    actual = [1] * 43 + [2] * 10 + [0] * 30
    data = {'Actual': actual, 'detect': result}
    df = pd.DataFrame(data)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(df['Actual'], df['detect'])

    # 将混淆矩阵转为 Pandas DataFrame
    conf_matrix = pd.DataFrame(conf_matrix, columns=['ok', 'daqipao', 'jiaodai'], index=['ok', 'daqipao', 'jiaodai'])

    # 打印混淆矩阵 DataFrame
    return conf_matrix

def cal_angle(angle_list):
    # 计算角度的均值和方差
    angle_ok = []
    angle_bubble = []
    angle_jiaodai = []
    for i in range(0, 83):
        if i < 43:
            # 大气泡
            angle_bubble.append(angle_list[i])
        elif i >= 43 and i < 53:
            # 胶带
            angle_jiaodai.append(angle_list[i])
        else:
            # ok
            angle_ok.append(angle_list[i])

    angle_all = []
    # ok均值
    angle_mean_ok = statistics.mean(angle_ok)
    angle_all.append(angle_mean_ok)
    # ok方差
    angle_variance_ok = statistics.variance(angle_ok)
    angle_all.append(angle_variance_ok)
    # 气泡均值
    angle_mean_bubble = statistics.mean(angle_bubble)
    angle_all.append(angle_mean_bubble)
    # 气泡方差
    angle_variance_bubble = statistics.variance(angle_bubble)
    angle_all.append(angle_variance_bubble)
    # 胶带均值
    angle_mean_jiaodai = statistics.mean(angle_jiaodai)
    angle_all.append(angle_mean_jiaodai)
    # 胶带方差
    angle_variance_jiaodai = statistics.variance(angle_jiaodai)
    angle_all.append(angle_variance_jiaodai)

    return angle_all

if __name__ == '__main__':
    path = 'D:/homework_design/data_test_tube'

    img_ok = cv2.imread(path + '/OK/OK_0001.bmp', 1)
    img = img_ok.copy()
    img_ok = cut(img_ok)
    img_ok, _ = rotate(img_ok)
    img_box, img_rect = cal_contours_box(img_ok)

    images = []
    hist_ok, _, _, _ = progress(img)

    all_hist_list = []
    length_list = []

    # 角度列表
    angle_list = []
    # iou列表
    iou_list = []

    # 遍历图片逐一检测
    for root, dirs, files in os.walk(path):
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            if img is not None:
                images.append(img)
            hist_img, length, angle, iou = progress(img)
            hist_img = sum(abs(hist_img - hist_ok))
            all_hist_list.append(hist_img)
            # min: 140 max: 211
            length_list.append((length))
            angle_list.append(angle)
            iou_list.append(iou)

    # draw_list = []
    # draw_list.append(min(length_list[0:43]))
    # draw_list.append(max(length_list[0:43]))
    # draw_list.append(min(length_list[43:53]))
    # draw_list.append(max(length_list[43:53]))
    # draw_list.append(min(length_list[53:83]))
    # draw_list.append(max(length_list[53:83]))
    # print(draw_list)
    # plt.title('contours_number')
    # plt.plot(draw_list)
    # plt.show()

    # 计算角度
    angle_all = cal_angle(angle_list)
    # print(angle_all)
    print("OK的角度均值为: {:.2f}".format(angle_all[0]), "OK的角度方差为: {:.2f}".format(angle_all[1]))
    print("daqipao的角度均值为: {:.2f}".format(angle_all[2]), "daqipao的角度方差为: {:.2f}".format(angle_all[3]))
    print("jiaodai的角度均值为: {:.2f}".format(angle_all[4]), "jiaodai的角度方差为: {:.2f}".format(angle_all[5]))

    # 计算iou查看对其效果
    all_iou = np.array(iou_list)
    print("平均IOU为: {:.2f}".format(all_iou.mean()))

    # 设置轮廓阈值
    min_contours = 135
    detect_result = detect(all_hist_list, length_list, min_contours)
    # print(detect_result)
    # print(len(detect_result))
    # 计算混淆矩阵
    confusion_matrix = draw_confusion_matrix(detect_result)
    print(confusion_matrix)
