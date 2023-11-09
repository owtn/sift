import json
import os
import random
import re
import shutil
import sys
from tqdm import tqdm
from glob import glob

import cv2 as cv
import easyocr
import numpy as np
import torch

from shapely.geometry import Polygon, Point


def dotted_line(img, xmin, ymin, xmax, ymax, seglen=7, spacelen=7, color=(0, 0, 255), thick=5):
    """
    draw_box()用到，在图中画一条虚线，只支持水平或竖直
    Args:
        img: numpy数组，绘制的原始图像
        xmin: 起始x坐标
        ymin: 起始y坐标
        xmax: 终点x坐标
        ymax: 终点y坐标
        seglen: 每段虚线长度
        spacelen: 每段虚线间隔
        color: 颜色
        thick: 粗细

    Returns:
    """
    if xmin == xmax:
        for y in range(ymin, ymax, seglen + spacelen):
            cv.line(img, (xmin, y), (xmin, min(ymax, y + seglen)), color, thick)
    if ymin == ymax:
        for x in range(xmin, xmax, seglen + spacelen):
            cv.line(img, (x, ymin), (min(xmax, x + seglen), ymin), color, thick)


# def draw_box(img, xmin, ymin, xmax, ymax, seglen=7, spacelen=7, color=(0,0,255), thick=5):
#     if spacelen == 0:
#         cv.line(img, (xmin, ymin), (xmin, ymax), color, thick)
#         cv.line(img, (xmin, ymin), (xmax, ymin), color, thick)
#         cv.line(img, (xmax, ymin), (xmax, ymax), color, thick)
#         cv.line(img, (xmin, ymax), (xmax, ymax), color, thick)
#     else:
#         dotted_line(img, xmin, ymin, xmin, ymax, seglen=seglen, spacelen=spacelen, color=color, thick=thick)
#         dotted_line(img, xmin, ymin, xmax, ymin, seglen=seglen, spacelen=spacelen, color=color, thick=thick)
#         dotted_line(img, xmax, ymin, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color, thick=thick)
#         dotted_line(img, xmin, ymax, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color, thick=thick)
def draw_box(img, xmin, ymin, xmax, ymax, seglen=7, expan=1, color1=(0, 0, 255), color2=(1, 1, 1), thick=5):
    """
    两种颜色交替的虚线包围框
    Args:
        img: 原始图像
        xmin: 起始x坐标
        ymin: 起始y坐标
        xmax: 终点x坐标
        ymax: 终点y坐标
        seglen: 每段虚线长度
        expan: 虚线间隔与线段长度之比，即space = expan * seglen
        color1: 颜色1
        color2: 颜色2
        thick: 粗细

    Returns:
    """
    spacelen = (expan * 2 + 1) * seglen
    offset = (expan + 1) * seglen
    if spacelen == 0:
        cv.line(img, (xmin, ymin), (xmin, ymax), color1, thick)
        cv.line(img, (xmin, ymin), (xmax, ymin), color1, thick)
        cv.line(img, (xmax, ymin), (xmax, ymax), color1, thick)
        cv.line(img, (xmin, ymax), (xmax, ymax), color1, thick)
    else:
        dotted_line(img, xmin, ymin, xmin, ymax, seglen=seglen, spacelen=spacelen, color=color1, thick=thick)
        dotted_line(img, xmin, ymin, xmax, ymin, seglen=seglen, spacelen=spacelen, color=color1, thick=thick)
        dotted_line(img, xmax, ymin, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color1, thick=thick)
        dotted_line(img, xmin, ymax, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color1, thick=thick)

        dotted_line(img, xmin, ymin + offset, xmin, ymax, seglen=seglen, spacelen=spacelen, color=color2, thick=thick)
        dotted_line(img, xmin + offset, ymin, xmax, ymin, seglen=seglen, spacelen=spacelen, color=color2, thick=thick)
        dotted_line(img, xmax, ymin + offset, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color2, thick=thick)
        dotted_line(img, xmin + offset, ymax, xmax, ymax, seglen=seglen, spacelen=spacelen, color=color2, thick=thick)


def get_subImage(outputPath):
    """
    读取json，获得子图列表
    Args:
        outputPath: json目录

    Returns:
        子图信息列表，[xmin, ymin, xmax, ymax, 子图类别]
    """
    sub_set = []
    try:
        with open(outputPath, encoding='utf-8') as load_f:
            data = json.load(load_f)
            for o in data['outputs']['object']:
                if o['name'] == '统计图' or o['name'] == '示意图' or o['name'] == '其他':
                    continue

                # 在最后添加了子图类别用于ocr剔除merge
                sub_set.append(
                    [o['bndbox']['xmin'], o['bndbox']['ymin'], o['bndbox']['xmax'], o['bndbox']['ymax'], o['name']])
    except:
        return []
    return sub_set


def point_in_squares(coord, squares):
    """
    判断点是否在若干个矩形的范围内，只支持水平竖直矩形
    Args:
        coord: 点坐标
        squares: 矩形列表，矩形描述：[a坐标，b坐标，c坐标，d坐标],a为左上点，abcd顺时针排列

    Returns:
        点在矩形内：true， 否则false
    """
    polygons = [Polygon(sq) for sq in squares]
    point = Point(coord[0], coord[1])
    # for square in squares:
    #     if square[0][0] <= coord[0] <= square[2][0] and square[0][1] <= coord[1] <= square[2][1]:
    #         return True
    for polygon in polygons:
        if polygon.intersects(point):
            return True
    return False


def sift_match(rgb1, rgb2, grey1, grey2, img1, img2, ocr_reader, T=0.7, rate=0.5, color=(0, 255, 0), useOCR=True,
               logfile=None, img_path=None):
    """
    计算两张子图的sift特征，并使用FLANN匹配
    Args:
        rgb1: 图1，numpy数组
        rgb2: 图2，numpy数组
        grey1: 图1灰度格式
        grey2: 图2灰度格式
        img1: 图1灰度转rgb，用来做ocr的输入
        img2: 图2灰度转rgb，用来做ocr的输入
        ocr_reader: EasyOcr模型
        T: 2nn阈值，次佳匹配与最佳匹配比值小于T才有效
        rate: 特征点有效比例阈值，有效匹配特征点数比例大于rate认为图1图2相似
        color: 匹配线颜色
        useOCR: 是否启用ocr过滤文字区域
        logfile: log文件，记录报错的图像
        img_path: 输入图像路径，用于logfile

    Returns:
        成果匹配返回匹配连线图，不成功返回None
    """
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    try:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    except Exception as e:
        kp1, des1, kp2, des2 = [], [], [], []
        print('detectAndCompute error,', e)
        if logfile is not None:
            with open(logfile, 'a') as f:
                f.write('\n' + str(e) + '\n')
                if img_path is not None:
                    f.write(img_path)
                f.write('\n')

    if des1 is not None and des2 is not None:

        match_len1 = len(kp1)
        match_len2 = len(kp2)
        if match_len1 < 3 or match_len2 < 3:
            return None
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        reverse_matches = flann.knnMatch(des2, des1, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # 匹配记录
        matchesMap = []

        # ratio test as per Lowe's paper
        count = 0
        for i, (m, n) in enumerate(reverse_matches):
            if m.distance < T * n.distance:
                ki1 = m.queryIdx
                ki2 = m.trainIdx
                matchesMap.append((ki1, ki2))
        for i, (m, n) in enumerate(matches):
            if m.distance < T * n.distance:
                ki1 = m.queryIdx
                ki2 = m.trainIdx
                if (ki2, ki1) in matchesMap:
                    matchesMask[i] = [1, 0]
                    count += 1

        #         DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        #         DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

        c1 = len(des1)
        c2 = len(des2)

        # 子图内ocr
        if count > rate * c1 and count > 4 and useOCR:
            # 子图ocr识别
            ocr_result1 = readtext(ocr_reader, grey1, img1)
            ocr_result2 = readtext(ocr_reader, grey2, img2)
            squares1 = [result[0] for result in ocr_result1]
            squares2 = [result[0] for result in ocr_result2]
            for i, (m, n) in enumerate(matches):
                if matchesMask[i][0] == 1:
                    if point_in_squares(kp1[m.queryIdx].pt, squares1) or point_in_squares(kp2[m.trainIdx].pt, squares2):
                        matchesMask[i] = [0, 0]
                        count -= 1

        if count > rate * c1 and count > 4:
            return cv.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches, None,
                                     matchColor=color,
                                     singlePointColor=(255, 0, 0),
                                     matchesMask=matchesMask,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return None


def readtext(reader, img, img_cv_grey, decoder='greedy', beamWidth=5, batch_size=1,
             workers=0, allowlist=None, blocklist=None, detail=1,
             rotation_info=None, paragraph=False, min_size=20,
             contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
             text_threshold=0.7, low_text=0.4, link_threshold=0.4,
             canvas_size=2560, mag_ratio=1.,
             slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
             width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, output_format='standard'):
    """
    ocr提取文字
    Args:
        reader: EasyOcr模型
        img: 输入rgb图
        img_cv_grey: 输入灰度图
        decoder: EasyOcr默认参数
        beamWidth: EasyOcr默认参数
        batch_size: EasyOcr默认参数
        workers: EasyOcr默认参数
        allowlist: EasyOcr默认参数
        blocklist: EasyOcr默认参数
        detail: 传入1时，返回文字矩形区域，文字，置信度，例如：
                ([[114, 338], [152, 338], [152, 346], [114, 346]], 'P00i4', 0.15424631062552452)
                传入0时仅返回文字列表
        rotation_info: EasyOcr默认参数
        paragraph: EasyOcr默认参数
        min_size: EasyOcr默认参数
        contrast_ths: EasyOcr默认参数
        adjust_contrast: EasyOcr默认参数
        filter_ths: EasyOcr默认参数
        text_threshold: EasyOcr默认参数
        low_text: EasyOcr默认参数
        link_threshold: EasyOcr默认参数
        canvas_size: EasyOcr默认参数
        mag_ratio: EasyOcr默认参数
        slope_ths: EasyOcr默认参数
        ycenter_ths: EasyOcr默认参数
        height_ths: EasyOcr默认参数
        width_ths: EasyOcr默认参数
        y_ths: EasyOcr默认参数
        x_ths: EasyOcr默认参数
        add_margin: EasyOcr默认参数
        output_format: EasyOcr默认参数

    Returns:
        检测结果列表，形式由参数detail控制
        传入1时，返回文字矩形区域，文字，置信度，例如：
        ([[114, 338], [152, 338], [152, 346], [114, 346]], 'P00i4', 0.15424631062552452)
        传入0时仅返回文字列表
    """
    horizontal_list, free_list = reader.detect(img, min_size, text_threshold,
                                               low_text, link_threshold,
                                               canvas_size, mag_ratio,
                                               slope_ths, ycenter_ths,
                                               height_ths, width_ths,
                                               add_margin, False)
    horizontal_list, free_list = horizontal_list[0], free_list[0]
    result = reader.recognize(img_cv_grey, horizontal_list, free_list,
                              decoder, beamWidth, batch_size,
                              workers, allowlist, blocklist, detail, rotation_info,
                              paragraph, contrast_ths, adjust_contrast,
                              filter_ths, y_ths, x_ths, False, output_format)
    return result


# 判断ocr识别结果中是否有merge
def haveMerge(ocr_list):
    """
    判断ocr识别结果中是否有类似merge含义的文字
    Args:
        ocr_list: readtext(detail=0)返回结果，文字列表

    Returns:
        存在merge含义返回true，不存在返回false
    """
    pattern = re.compile(r'.*(?:a|w|m|v|hl)[eoc][vrt][jnge][reoc]d?|.*ov[eoc]rlay|.*[od0l][o4a]p[il1]', re.I)
    for word in ocr_list:
        if not (pattern.match(word) is None):
            return True
    return False


def IFD(ocr_reader, to_be_detected_file_path, res_period_path, T, rate, useOCR=True):
    """
    对一批数据进行检测，目录结构：
    根目录
    |---期刊1
        |---大图1
        |---json1
        |---大图2
        |---json2

    Args:
        ocr_reader: EasyOcr模型
        to_be_detected_file_path: 待检测目录
        res_period_path: 结果输出目录
        T: sift_match()参数T
        rate: sift_match()参数rate
        useOCR: 是否启用ocr

    Returns:
        total_num: 检测大图总数
        res_num: 有问题大图总数
        part_img_num: 有问题子图总数
    """
    if not os.path.exists(to_be_detected_file_path):
        print("该路径不存在：" + to_be_detected_file_path)
        return
    paper_list = os.listdir(to_be_detected_file_path)
    paper_list.sort()
    total_num = len(paper_list)
    res_num = 0
    log_file = os.path.join(res_period_path, 'log.txt')
    progress_file = os.path.join(res_period_path, 'progress.txt')
    part_img_num = 0

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            log_data = f.read()
        finish_papers = re.findall("论文完成检测:(.+?)\n", log_data)
        if len(finish_papers) > 0:
            last_paper = finish_papers[-1]
            last_index = paper_list.index(last_paper)
            paper_list = paper_list[last_index + 1:]
    for paper in tqdm(paper_list):
        res_paper_path = os.path.join(res_period_path, paper)
        imgPaths = glob(os.path.join(to_be_detected_file_path, paper, '*.jpg')) + \
                   glob(os.path.join(to_be_detected_file_path, paper, '*.png'))
        result_flag = False
        for imgPath in imgPaths:
            hashCode = '.'.join(imgPath.split('/')[-1].split('.')[:-1])
            outputPath = os.path.join(to_be_detected_file_path, paper, hashCode + '.json')
            if not os.path.exists(outputPath):
                with open(log_file, 'a+') as f:
                    f.write('\n' + imgPath + '\n缺少子图标注文件\n')
                print('图片', imgPath, '缺少子图标注文件')
                continue
            # OCR检测结果
            img_have_merge = False
            try:
                img_rgb = cv.imread(imgPath)
                ocr_grey = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
                ocr_image = cv.cvtColor(ocr_grey, cv.COLOR_GRAY2RGB)
            except:
                print('读图失败！')
                print('图片路径：', imgPath)
                with open(log_file, 'a+') as f:
                    f.write('\n' + imgPath + '\n读图失败\n')
                continue
            if useOCR:
                ocr_results = readtext(ocr_reader, ocr_image, ocr_grey, detail=0)
                # ocr_image_left90 = np.rot90(ocr_image, k=1, axes=(0, 1))
                # ocr_grey_left90 = np.rot90(ocr_grey, k=1)
                # ocr_results += readtext(ocr_reader, ocr_image_left90, ocr_grey_left90, detail=0)
                ocr_image_right90 = np.rot90(ocr_image, k=3, axes=(0, 1))
                ocr_grey_right90 = np.rot90(ocr_grey, k=3)
                ocr_results += readtext(ocr_reader, ocr_image_right90, ocr_grey_right90, detail=0)
                if haveMerge(ocr_results):
                    img_have_merge = True
            img = ocr_grey
            if img is None or len(img) == 0:
                continue
            sub_set = get_subImage(outputPath)
            n = len(sub_set)
            part_img_num += n
            img_full = img_rgb.copy()
            for i in range(0, n - 1):
                p1 = sub_set[i]
                p1 = [max(cord, 0) for cord in p1[:4]]
                if p1[3] <= p1[1] or p1[2] <= p1[0]:
                    continue
                # 有merge不检测染色图
                if img_have_merge and sub_set[i][4] == '染色图':
                    continue
                for j in range(i + 1, n):
                    # 有merge不检测染色图
                    if img_have_merge and sub_set[j][4] == '染色图':
                        continue
                    p2 = sub_set[j]
                    p2 = [max(cord, 0) for cord in p2[:4]]
                    if p2[3] <= p2[1] or p2[2] <= p2[0]:
                        continue
                    img1 = img[p1[1]:p1[3], p1[0]:p1[2]]
                    grey2rgb1 = ocr_image[p1[1]:p1[3], p1[0]:p1[2]]
                    img2 = img[p2[1]:p2[3], p2[0]:p2[2]]
                    grey2rgb2 = ocr_image[p2[1]:p2[3], p2[0]:p2[2]]
                    rgb1 = img_rgb[p1[1]:p1[3], p1[0]:p1[2]]
                    rgb2 = img_rgb[p2[1]:p2[3], p2[0]:p2[2]]
                    rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    pink_color = (100, 100, 255)
                    pink_color = (128, 255, 255)
                    pink_color = (192, 192, 255)
                    red_color = (0, 0, 255)
                    black_color = (0, 0, 0)
                    dot_line_param = {'seglen': 6, 'expan': 2}
                    img3 = sift_match(rgb1, rgb2, grey2rgb1, grey2rgb2, img1, img2, ocr_reader, T=T, rate=rate,
                                      color=red_color,
                                      useOCR=useOCR, logfile=log_file, img_path=imgPath)
                    if img3 is not None:
                        if not os.path.exists(res_paper_path):
                            os.makedirs(res_paper_path)
                            res_num = res_num + 1
                            result_flag = True
                        shutil.copy(imgPath, res_paper_path)
                        matchName = hashCode + '_match' + str(i) + '&' + str(j)
                        cv.imwrite(os.path.join(res_paper_path, matchName + '_area.jpg'), img3)
                        img_retangle = img_rgb.copy()
                        img_redbox = img_rgb.copy()
                        # cv.rectangle(img_retangle, (p1[0], p1[1]), (p1[2], p1[3]), red_color, 2)
                        # cv.rectangle(img_retangle, (p2[0], p2[1]), (p2[2], p2[3]), red_color, 2)
                        # cv.rectangle(img_full, (p1[0], p1[1]), (p1[2], p1[3]), rand_color, 2)
                        # cv.rectangle(img_full, (p2[0], p2[1]), (p2[2], p2[3]), rand_color, 2)
                        draw_box(img_retangle, p1[0], p1[1], p1[2], p1[3], seglen=dot_line_param['seglen'],
                                 expan=dot_line_param['expan'], color1=black_color, color2=pink_color, thick=5)
                        draw_box(img_retangle, p2[0], p2[1], p2[2], p2[3], seglen=dot_line_param['seglen'],
                                 expan=dot_line_param['expan'], color1=black_color, color2=pink_color, thick=5)
                        draw_box(img_full, p1[0], p1[1], p1[2], p1[3], seglen=dot_line_param['seglen'],
                                 expan=dot_line_param['expan'], color1=rand_color, color2=rand_color, thick=5)
                        draw_box(img_full, p2[0], p2[1], p2[2], p2[3], seglen=dot_line_param['seglen'],
                                 expan=dot_line_param['expan'], color1=rand_color, color2=rand_color, thick=5)
                        cv.imwrite(os.path.join(res_paper_path, matchName + '.jpg'), img_retangle)
                        cv.imwrite(os.path.join(res_paper_path, hashCode + '_full.jpg'), img_full)
        with open(progress_file, 'a+') as f:
            f.write('\n' + '论文完成检测:' + paper + '\n')
            if result_flag:
                f.write('此论文检出复用！\n')
    return total_num, res_num, part_img_num


def detect(detectPath, resPath, T, rate, logger=None, useOCR=True):
    """
    初始化ocr，调用IFD()对检测目录进行检测
    Args:
        logger: logger
        detectPath: 要检测的数据根目录
        resPath: 结果输出目录
        T: sift_match参数T
        rate: sift_match参数rate
        useOCR: 是否启用ocr

    Returns:
    """
    if logger is not None:
        logger.log('开始检测' + detectPath + f', 参数：[T:{T}, rate:{rate}]')
    log_file = os.path.join(resPath, 'log.txt')
    if not os.path.exists(detectPath):
        print("该路径不存在：" + detectPath)
        with open(log_file, 'a+') as f:
            f.write('\n' + detectPath + '该路径不存在\n')
        return
    if not os.path.exists(resPath):
        os.makedirs(resPath)
    if useOCR:
        # OCR识别文字工具
        ocr_reader = easyocr.Reader(['en'], gpu=True)
        ocr_reader.detector.eval()
        ocr_reader.recognizer.eval()
    else:
        ocr_reader = None
    total_num, res_num, part_img_num = IFD(ocr_reader, detectPath, resPath, T, rate, useOCR=useOCR)
    print("【本次共检测{0}篇论文{1}张子图，其中{2}篇论文疑似存在造假图片.】".format(total_num, part_img_num, res_num))
    if logger is not None:
        logger.log('检测完成！')


if __name__ == '__main__':
    year = sys.argv[1]
    batch = sys.argv[2]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    detect_path = f"/home/hdd1/data/wanfang/{year}/IFD{year}/{batch}"
    res_path = f"/home/hdd1/data/wanfang/{year}/sift_v4/{batch}"
    detect(detect_path, res_path, 0.65, 0.55, useOCR=True)
    # detect(detect_path, res_path, 0.8, 0.3, useOCR=True)
