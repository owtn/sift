from glob import glob
import os
import cv2 as cv

def  sift_match(rgb1, rgb2, img1, img2, ocr_reader, T=0.7, rate=0.5, color=(0, 255, 0), useOCR=True,
               logfile=None, img_path=None):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    try:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    except Exception as e:
        kp1, des1, kp2, des2 = [], [], [], []
        print('detectAndCompute error,', e)
        print(img_path)



    if des1 is not None and des2 is not None:

        match_len1 = len(kp1)
        match_len2 = len(kp2)
        special = False
        if '/home/hdd1/wanghaoran/sift/data/solo/jietu/6-4' in img_path:
            special = True
            print(match_len1, match_len2)
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


        if '/home/hdd1/wanghaoran/sift/data/solo/jietu/6-4' in img_path:
            print(c1, c2, count)
        if count > rate * c1 and count > 0:
            return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                                     matchColor=color,
                                     singlePointColor=(255, 0, 0),
                                     matchesMask=matchesMask,
                                     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return None

if __name__ == '__main__':
    base_path = '/home/hdd1/wanghaoran/sift/data/solo/jietu'
    dirs = glob(os.path.join(base_path, '*'))
    for dir in dirs:
        img_paths = glob(os.path.join(dir, '*.png'))
        if len(img_paths) < 2:
            print(dir, 'less than 2 imgs')
            continue
        path1, path2 = img_paths[:2]
        img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)
        rgb1 = cv.imread(path1)
        img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)
        rgb2 = cv.imread(path2)
        img3 = sift_match(rgb1, rgb2, img1, img2, None, 0.9, 0, useOCR=False, img_path=path1+'&'+path2)
        if img3 is not None:
            save_dir = os.path.join(dir, 'result')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv.imwrite(os.path.join(save_dir, 'result.jpg'), img3)