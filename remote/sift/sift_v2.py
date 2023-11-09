import json
import os
import random
import shutil
import sys
from tqdm import tqdm

import cv2 as cv
from glob import glob

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

def get_subImage(outputPath):
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

# 计算sift特征，并使用FLANN做匹配
# 有两个可调参数，T为2nn的阈值;rate为mask个数与matches个数的比值
def sift_match(rgb1, rgb2, img1, img2, T=0.7, rate=0.5, color=(0, 255, 0)):
    sift = cv.SIFT_create()
    try:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    except Exception as e:
        kp1, des1, kp2, des2 = [], [], [], []
        print('detectAndCompute error,', e)

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


        if count > rate * c1 and count > 4:
            return cv.drawMatchesKnn(rgb1, kp1, rgb2, kp2, matches, None,
                                     matchColor=color,
                                     singlePointColor=(255, 0, 0),
                                     matchesMask=matchesMask,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return None

def IFD(to_be_detected_file_path, res_period_path, T, rate):
    if not os.path.exists(to_be_detected_file_path):
        print("该路径不存在：" + to_be_detected_file_path)
        return
    paper_list = os.listdir(to_be_detected_file_path)
    paper_list.sort()
    total_num = len(paper_list)
    res_num = 0
    log_file = os.path.join(res_period_path, 'log.txt')
    part_img_num = 0

    for paper in tqdm(paper_list):
        res_paper_path = os.path.join(res_period_path, paper)
        imgPaths = glob(os.path.join(to_be_detected_file_path, paper, '*.jpg')) + \
                   glob(os.path.join(to_be_detected_file_path, paper, '*.png'))
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
                    img3 = sift_match(rgb1, rgb2, img1, img2, T=T, rate=rate, color=red_color)
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
    return total_num, res_num, part_img_num


def detect(detectPath, resPath, T, rate):
    if not os.path.exists(detectPath):
        print("该路径不存在：" + detectPath)
        return
    total_num, res_num, part_img_num = IFD(detectPath, resPath, T, rate)
    print("【本次共检测{0}篇论文，其中{1}篇论文疑似存在造假图片.】".format(total_num, res_num))


if __name__ == '__main__':
    year = sys.argv[1]
    batch = sys.argv[2]
    detect_path = f"/home/hdd1/data/wanfang/{year}/IFD{year}/{batch}"
    res_path = f"/home/hdd1/data/wanfang/{year}/sift_v2/{batch}"
    detect(detect_path, res_path, 0.7, 0.5)
