import json
import os
import random
import re
import shutil

import cv2 as cv
import easyocr
import numpy as np


# 读取图像的标注文件，对于每张图片进行子图提取
def get_subImage(img_name, outputs_path):
    sub_set = []
    try:
        with open(outputs_path + img_name + '.json', encoding='utf-8') as load_f:
            data = json.load(load_f)
            for o in data['outputs']['object']:
                if o['name'] == '统计图' or o['name'] == '示意图' or o['name'] == '其他' or o['bndbox']['xmin'] < 0 or \
                        o['bndbox']['ymin'] < 0 or o['bndbox']['xmax'] > data['size']['width'] or o['bndbox']['ymax'] > \
                        data['size']['height'] or o['bndbox']['xmin'] == o['bndbox']['xmax'] or o['bndbox']['ymin'] == \
                        o['bndbox']['ymax']:
                    continue

                # 在最后添加了子图类别用于ocr剔除merge
                sub_set.append(
                    [o['bndbox']['xmin'], o['bndbox']['ymin'], o['bndbox']['xmax'], o['bndbox']['ymax'], o['name']])
    except:
        return []
    return sub_set

#检查特征点点坐标是否在文本区域内
def point_in_squares(coord, squares):
    for square in squares:
        if square[0][0] <= coord[0] <= square[2][0] and square[0][1] <= coord[1] <= square[2][1]:
            return True
    return False

# 计算sift特征，并使用FLANN做匹配
# 有两个可调参数，T为2nn的阈值;rate为mask个数与matches个数的比值
def sift_match(ii, jj, rgb1, rgb2, img1, img2, ocr_reader, T=0.7, rate=0.5, color=(0, 255, 0)):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    #    print('kp1[0]:',kp1[0])
    #    print('kp1[0].pt', kp1[0].pt)
    #     print(type(des1))
    #     print(type(des2))
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

        # 特征点对坐标
        coords = []

        # 匹配记录
        matchesMap = []
        # debugmap = []

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
                #if (ki2, ki1) in matchesMap:
                    matchesMask[i] = [1, 0]
                    # debugmap.append((ki2, ki1))
                    count += 1
                    #coords.append((kp1[ki1].pt, kp2[ki2].pt))

        #         DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        #         DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

        c1 = len(des1)
        c2 = len(des2)
        if ii == 6 and jj == 7:
            print(matchesMap)
            print('count', count)
            print('c1', c1)
            print('c2', c2)
            print('points', [kp1[i].pt for i in range(len(kp1)) if matchesMask[i][0] == 1])
        if count > rate * min(c1, c2) and count > 4:
            # 子图ocr识别
            ocr_result1 = readtext(ocr_reader, rgb1, img1)
            ocr_result2 = readtext(ocr_reader, rgb2, img2)
            squares1 = [result[0] for result in ocr_result1]
            squares2 = [result[0] for result in ocr_result2]
            for i, (m, n) in enumerate(matches):
                if matchesMask[i][0] == 1:
                    if point_in_squares(kp1[m.queryIdx].pt, squares1) or point_in_squares(kp2[m.trainIdx].pt, squares2):
                        matchesMask[i] = [0,0]
                        count -= 1

            if ii == 6 and jj == 7:
                print(matchesMap)
                print(count)
                print('c1', c1)
                print('c2', c2)
                print('squares1:', squares1[0])
                print('points', [kp1[i].pt for i in range(len(kp1)) if matchesMask[i][0] == 1])
        if count > rate * min(c1, c2) and count > 4:
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                 matchColor=color,
                                 singlePointColor=(255, 0, 0),
                                 matchesMask=matchesMask,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return None


# ocr提取文字
def readtext(reader, img, img_cv_grey, decoder='greedy', beamWidth=5, batch_size=1,
             workers=0, allowlist=None, blocklist=None, detail=1,
             rotation_info=None, paragraph=False, min_size=20,
             contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
             text_threshold=0.7, low_text=0.4, link_threshold=0.4,
             canvas_size=2560, mag_ratio=1.,
             slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
             width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, output_format='standard'):
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


def judgeTime(file, dateStr):
    return True


# 判断ocr识别结果中是否有merge
def haveMerge(ocr_list):
    pattern = re.compile(r'.*(?:a|w|m|v|hl)[eoc][vrt][jnge][reoc]d?|.*ov[eoc]rlay|.*[od0l][o4a]p[il1]', re.I)
    for word in ocr_list:
        if not (pattern.match(word) is None):
            return True
    return False



def IFD(to_be_detected_file_path, T, rate):
    global use_ocr
    if not os.path.exists(to_be_detected_file_path):
        print("该路径不存在：" + to_be_detected_file_path)
        return
    paper_list = os.listdir(to_be_detected_file_path)
    # res_period_path = to_be_detected_file_path.replace('/data/','/result/')
    res_period_path = to_be_detected_file_path
    no = 0
    res_no = 0

    paper_name = '111'

    # OCR识别工具
    if use_ocr:
        ocr_reader = easyocr.Reader(['en'])
    else:
        ocr_reader = None

    for paper in paper_list:
        if judgeTime(to_be_detected_file_path + '/' + paper, "2021-12-13"):
            no = no + 1
            print(str(no) + ':' + paper + ', ', end="")
            res_paper_path = res_period_path + '/' + paper + '/'
            paper_path = to_be_detected_file_path + '/' + paper + '/paper/'
            img_path = to_be_detected_file_path + '/' + paper + '/images/'
            outputs_path = to_be_detected_file_path + '/' + paper + '/outputs/'
            if not os.path.exists(outputs_path):
                print("该论文无标注文件.")
                continue
            if not os.path.exists(img_path):
                print("该论文无图像.")
                continue
            img_list = os.listdir(img_path)
            for img_name in img_list:

                if img_name[-3:] == 'png' or img_name[-3:] == 'jpg':

                    # OCR检测结果
                    img_have_merge = False
                    ocr_path = img_path + img_name
                    # ocr_image = loadImage(ocr_path)
                    img_rgb = cv.imread(img_path + img_name)
                    ocr_image = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
                    # try:
                    #    img_rgb = cv.imread(img_path+img_name)
                    #    ocr_image = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
                    # except:
                    #    print("===============加载图片错误==============")
                    #    continue
                    ocr_grey = cv.imread(ocr_path, cv.IMREAD_GRAYSCALE)
                    if use_ocr:
                        ocr_results = readtext(ocr_reader, ocr_image, ocr_grey, detail=0)
                        ocr_image_left90 = np.rot90(ocr_image, k=1, axes=(0, 1))
                        ocr_grey_left90 = np.rot90(ocr_grey, k=1)
                        ocr_results += readtext(ocr_reader, ocr_image_left90, ocr_grey_left90, detail=0)
                        ocr_image_right90 = np.rot90(ocr_image, k=3, axes=(0, 1))
                        ocr_grey_right90 = np.rot90(ocr_grey, k=3)
                        ocr_results += readtext(ocr_reader, ocr_image_right90, ocr_grey_right90, detail=0)
                        # print([elem[1] for elem in ocr_results])
                        if haveMerge(ocr_results):
                            img_have_merge = True

                    print('----------------------------------')
                    print(img_name)
                    # print(ocr_results)

                    img = ocr_grey
                    sub_set = get_subImage(img_name[:-4], outputs_path)
                    n = len(sub_set)
                    c = 1
                    for i in range(0, n - 1):

                        # 有merge不检测染色图
                        if img_have_merge and sub_set[i][4] == '染色图':
                            continue
                        for j in range(i + 1, n):

                            # 有merge不检测染色图
                            if img_have_merge and sub_set[j][4] == '染色图':
                                continue
                            p1 = sub_set[i]
                            p2 = sub_set[j]
                            img1 = img[p1[1]:p1[3], p1[0]:p1[2]]
                            rgb1 = ocr_image[p1[1]:p1[3], p1[0]:p1[2]]
                            img2 = img[p2[1]:p2[3], p2[0]:p2[2]]
                            rgb2 = ocr_image[p2[1]:p2[3], p2[0]:p2[2]]
                            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                            img3 = sift_match(i, j, rgb1, rgb2, img1, img2, ocr_reader, T=T, rate=rate, color=color)
                            if img3 is not None:
                                if not os.path.exists(res_paper_path):
                                    os.makedirs(res_paper_path)
                                res_no = res_no + 1
                                shutil.copy(img_path + img_name, res_paper_path)
                                if os.path.exists(paper_path + paper + '.pdf') and not os.path.exists(
                                        res_paper_path + paper + '.pdf'):
                                    shutil.copy(paper_path + paper + '.pdf', res_paper_path)
                                elif os.path.exists(paper_path + paper + '.html') and not os.path.exists(
                                        res_paper_path + paper_name + '.html'):
                                    shutil.copy(paper_path + paper + '.html', res_paper_path)
                                cv.imwrite(res_paper_path + img_name[:-4] + '_' + str(i) + '&' + str(j) + '.jpg', img3)
                                c += 1
                                cv.rectangle(img_rgb, (p1[0], p1[1]), (p1[2], p1[3]), color, 2)
                                cv.rectangle(img_rgb, (p2[0], p2[1]), (p2[2], p2[3]), color, 2)
                                cv.imwrite(res_paper_path + img_name[:-4] + '_full.jpg', img_rgb)
                                # print('====================')
                                # print(res_paper_path+img_name[:-4]+'_full.jpg')
            print("该论文检测完毕.")
    print("【共检测{0}篇论文，其中{1}篇论文疑似存在造假图片.】".format(no, res_no))


use_ocr = True
IFD('/home/hdd1/wanghaoran/sift/data/combo', 0.7, 0.5)
