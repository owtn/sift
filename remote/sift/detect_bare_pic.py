import os
from glob import glob
import random
import cv2 as cv

def sift_match(img1, img2, T=0.7, rate=0.5, color=(0, 255, 0)):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
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
        # 子图ocr识别
        # ocr_result1 = readtext(ocr_reader, rgb1, img1)
        # ocr_result2 = readtext(ocr_reader, rgb2, img2)

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
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                 matchColor=color,
                                 singlePointColor=(255, 0, 0),
                                 matchesMask=matchesMask,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return None

def detect(in_path, out_path, T, rate):
    if not os.path.exists(in_path):
        print("该路径不存在：" + in_path)
        return
    img_list = glob(os.path.join(in_path, '*.jpg')) + glob(os.path.join(in_path, '*.png'))
    img_list.sort()
    pic_num = len(img_list)
    res_num = 0
    for i in range(pic_num - 1):
        img1 = cv.imread(img_list[i])
        img_rbg1 = cv.imread(img_list[i], 0)
        for j in range(i + 1, pic_num):
            img2 = cv.imread(img_list[j])
            img_rbg2 = cv.imread(img_list[j], 0)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img3 = sift_match(img1, img2, T=T, rate=rate, color=color)
            if img3 is not None:
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                res_num = res_num + 1
                cv.imwrite(os.path.join(out_path, str(i) + '_' + str(j)), img3)
    print("共检测出{0}对相似图片".format(res_num))
    return

if __name__ == '__main__':
    input_path = '/home/hdd1/wanghaoran/sift/data/temp/in'
    output_path = '/home/hdd1/wanghaoran/sift/data/temp/out'
    detect(input_path, output_path, 0.9, 0.3)