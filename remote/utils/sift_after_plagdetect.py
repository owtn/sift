import cv2 as cv
import numpy as np
import easyocr
import re
from shapely.geometry import Polygon, Point


def sift_after_plag(query_small_img: np.array, database_small_img_inform: dict, useOCR: bool = False, T=0.65, rate=0.55, match_graph=False):
    """
    接在剽窃检测后面的sift
    Args:
        query_small_img: 查询的小图，numpy数组
        database_small_img_inform: 数据库中子图信息，dict，格式如下：
            {
                "big_img_path": "",
                "classification": "染色图",
                "paper": "d0a052ecb86f4749ac9b43063525ed81",
                "big_img": "",
                "xmin": 536,
                "xmax": 708,
                "ymin": 774,
                "ymax": 917
            }
        useOCR: 是否启用OCR检测merge类关键字
        T: 灵敏度参数，越大越敏感
        rate: 灵敏度参数，越小越敏感

    Returns: True（sift判断为复用）/ False（sift判断不是复用）

    """
    try:
        database_big_img_gray = cv.imread(database_small_img_inform['big_img_path'], cv.IMREAD_GRAYSCALE)
        database_big_img_gray_rgb = cv.cvtColor(database_big_img_gray, cv.COLOR_GRAY2RGB)
    except:
        print('读图失败！')
        print('图片路径：', database_small_img_inform['big_img_path'])
        return False
    img_have_merge = False
    if useOCR:
        ocr_reader = OcrReader()
        ocr_results = readtext(ocr_reader, database_big_img_gray_rgb, database_big_img_gray, detail=0)
        ocr_image_right90 = np.rot90(database_big_img_gray_rgb, k=3, axes=(0, 1))
        ocr_gray_right90 = np.rot90(database_big_img_gray, k=3)
        ocr_results += readtext(ocr_reader, ocr_image_right90, ocr_gray_right90, detail=0)
        if have_merge(ocr_results):
            img_have_merge = True

    # 有merge不检测染色图
    if img_have_merge and database_small_img_inform['classification'] == '染色图':
        return False
    gray1 = cv.cvtColor(query_small_img, cv.COLOR_RGB2GRAY)
    gray2rgb1 = cv.cvtColor(gray1, cv.COLOR_GRAY2RGB)
    xmin, xmax, ymin, ymax = database_small_img_inform['xmin'], database_small_img_inform['xmax'], \
                             database_small_img_inform['ymin'], database_small_img_inform['ymax']
    gray2 = database_big_img_gray[ymin: ymax, xmin:xmax]
    gray2rgb2 = database_big_img_gray_rgb[ymin: ymax, xmin: xmax]
    result = sift_match(gray1 = gray2rgb1, gray2=gray2rgb2, img1=gray1, img2=gray2, T=T, rate=rate,
                      useOCR=useOCR, match_graph=match_graph)
    return result

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
def have_merge(ocr_list):
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


def sift_match(gray1, gray2, img1, img2, match_graph=False, T=0.7, rate=0.5, useOCR=True):
    """
    计算两张子图的sift特征，并使用FLANN匹配
    Args:
        gray1: 图1灰度转rgb，用来做ocr的输入
        gray2: 图2灰度转rgb，用来做ocr的输入
        img1: 图1灰度
        img2: 图2灰度
        T: 2nn阈值，次佳匹配与最佳匹配比值小于T才有效
        rate: 特征点有效比例阈值，有效匹配特征点数比例大于rate认为图1图2相似
        useOCR: 是否启用ocr过滤文字区域
        match_graph: =True时返回特征点匹配图，=False时返回是否复用

    Returns:
        返回是否复用True/false，或者特征点匹配图/None
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
            ocr_reader = OcrReader()
            # 子图ocr识别
            ocr_result1 = readtext(ocr_reader, gray1, img1)
            ocr_result2 = readtext(ocr_reader, gray2, img2)
            squares1 = [result[0] for result in ocr_result1]
            squares2 = [result[0] for result in ocr_result2]
            for i, (m, n) in enumerate(matches):
                if matchesMask[i][0] == 1:
                    if point_in_squares(kp1[m.queryIdx].pt, squares1) or point_in_squares(kp2[m.trainIdx].pt, squares2):
                        matchesMask[i] = [0, 0]
                        count -= 1

        if count > rate * c1 and count > 4:
            if not match_graph:
                return True
            img = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, None,
                              matchColor=(255,0,0),
                              singlePointColor=(255, 0, 0),
                              matchesMask=matchesMask,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite('/home/hdd1/wanghaoran/playground/debug/1.png', img)
            return img
    return None if match_graph else False


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


class OcrReader:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            ocr_reader = easyocr.Reader(['en'], gpu=True)
            ocr_reader.detector.eval()
            ocr_reader.recognizer.eval()
            cls.__instance = ocr_reader
            return ocr_reader
        else:
            return cls.__instance


if __name__ == '__main__':
    img_path = '/home/hdd1/data/wanfang/2020/IFD2020/0609/Cancer Imaging/ecc2af107810df5013bf9430913230b6/images/2519c517f1c4d4adb812df47251a548b.jpg'
    db_inform = {
                "big_img_path": '/home/hdd1/data/wanfang/2020/IFD2020/0609/Cancer Imaging/ecc2af107810df5013bf9430913230b6/images/2519c517f1c4d4adb812df47251a548b.jpg',
                "classification": "造影图",
                "paper": "d0a052ecb86f4749ac9b43063525ed81",
                "big_img": "",
                "xmin": 0,
                "xmax": 342,
                "ymin": 1,
                "ymax": 259
            }
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    query_small = img[1:259, 0:342]
    # cv.imwrite('/home/hdd1/wanghaoran/playground/debug/1.png', query_small)
    print(sift_after_plag(query_small, db_inform, useOCR=True))
