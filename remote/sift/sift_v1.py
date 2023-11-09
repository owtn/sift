import json
import os
import random
import shutil
import sys
from tqdm import tqdm

import cv2 as cv


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
                sub_set.append([o['bndbox']['xmin'], o['bndbox']['ymin'], o['bndbox']['xmax'], o['bndbox']['ymax']])
    except:
        return []
    return sub_set


# 计算sift特征，并使用FLANN做匹配
# 有两个可调参数，T为2nn的阈值;rate为mask个数与matches个数的比值
def sift_match(img1, img2, T=0.7, rate=0.5, color=(0, 255, 0)):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is not None and des2 is not None:
        if len(des1) < 3 or len(des2) < 3:
            return None
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        count = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < T * n.distance:
                matchesMask[i] = [1, 0]
                count += 1
        #         DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        #         DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                 matchColor=color,
                                 singlePointColor=(255, 0, 0),
                                 matchesMask=matchesMask,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        c1 = len(des1)
        c2 = len(des2)

        if count > rate * c1 and count > 4:
            return img3
    return None


def IFD(to_be_detected_file_path, res_period_path, T, rate):
    if not os.path.exists(to_be_detected_file_path):
        print("该路径不存在：" + to_be_detected_file_path)
        return
    paper_list = os.listdir(to_be_detected_file_path)
    total_num = 0
    res_num = 0
    for paper in paper_list:
        total_num = total_num + 1
        #print(str(total_num) + ':' + paper + ', ', end="")
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
                try:
                    img_rgb = cv.imread(img_path + img_name)
                    img = cv.imread(img_path + img_name, 0)
                except:
                    print('读图失败！')
                    print('图片路径：', img_path + img_name)
                    continue
                if img is None or len(img) == 0:
                    continue
                sub_set = get_subImage(img_name[:-4], outputs_path)
                n = len(sub_set)
                c = 1
                for i in range(0, n - 1):
                    p1 = sub_set[i]
                    if p1[3] <= p1[1] or p1[2] <= p1[0]:
                        continue
                    for j in range(i + 1, n):
                        p2 = sub_set[j]
                        if p2[3] <= p2[1] or p2[2] <= p2[0]:
                            continue
                        img1 = img[p1[1]:p1[3], p1[0]:p1[2]]
                        img2 = img[p2[1]:p2[3], p2[0]:p2[2]]
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        img3 = sift_match(img1, img2, T=T, rate=rate, color=color)
                        if img3 is not None:
                            if not os.path.exists(res_paper_path):
                                os.makedirs(res_paper_path)
                                res_num = res_num + 1
                            shutil.copy(img_path + img_name, res_paper_path)
                            if os.path.exists(paper_path + paper + '.pdf') and not os.path.exists(
                                    res_paper_path + paper + '.pdf'):
                                shutil.copy(paper_path + paper + '.pdf', res_paper_path)
                            elif os.path.exists(paper_path + paper + '.html') and not os.path.exists(
                                    res_paper_path + paper + '.html'):
                                shutil.copy(paper_path + paper + '.html', res_paper_path)
                            cv.imwrite(res_paper_path + img_name[:-4] + '_' + str(i) + '&' + str(j) + '.jpg', img3)
                            c += 1
                            cv.rectangle(img_rgb, (p1[0], p1[1]), (p1[2], p1[3]), color, 2)
                            cv.rectangle(img_rgb, (p2[0], p2[1]), (p2[2], p2[3]), color, 2)
                            cv.imwrite(res_paper_path + img_name[:-4] + '_full.jpg', img_rgb)
        #print("该论文检测完毕.")
    #print("【本期共检测{0}篇论文，其中{1}篇论文疑似存在造假图片.】".format(total_num, res_num))
    return total_num, res_num


def detect(detectPath, resPath, T, rate):
    total_num, res_num = 0, 0
    if not os.path.exists(detectPath):
        print("该路径不存在：" + detectPath)
        return
    journalList = os.listdir(detectPath)
    for journal in tqdm(journalList):
        volumeList = os.listdir(os.path.join(detectPath, journal))
        for volume in volumeList:
            num1, num2 = IFD(os.path.join(detectPath, journal, volume),
                             os.path.join(resPath, journal, volume), T, rate)
            total_num += num1
            res_num += num2
    print("【本次共检测{0}篇论文，其中{1}篇论文疑似存在造假图片.】".format(total_num, res_num))


if __name__ == '__main__':
    year = sys.argv[1]
    batch = sys.argv[2]
    detect_path = f"/home/hdd1/data/wanfang/{year}/IFD{year}/{batch}"
    res_path = f"/home/hdd1/data/wanfang/{year}/sift_v1/{batch}"
    detect(detect_path, res_path, 0.7, 0.5)
