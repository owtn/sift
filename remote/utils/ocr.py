import easyocr

from sift.sift_v3 import readtext
from sift.sift_v3 import haveMerge
import os
from glob import glob
import cv2 as cv
import numpy as np


def lr90_ocr(ocr_reader, ocr_path):
    img_rgb = cv.imread(ocr_path)
    ocr_image = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    ocr_grey = cv.imread(ocr_path, cv.IMREAD_GRAYSCALE)
    ocr_image = cv.cvtColor(ocr_grey, cv.COLOR_GRAY2RGB)
    results = readtext(ocr_reader, ocr_image, ocr_grey, detail=0)
    ocr_image_left90 = np.rot90(ocr_image, k=1, axes=(0, 1))
    ocr_grey_left90 = np.rot90(ocr_grey, k=1)
    left_results = readtext(ocr_reader, ocr_image_left90, ocr_grey_left90, detail=0)
    ocr_image_right90 = np.rot90(ocr_image, k=3, axes=(0, 1))
    ocr_grey_right90 = np.rot90(ocr_grey, k=3)
    right_results = readtext(ocr_reader, ocr_image_right90, ocr_grey_right90, detail=0)
    return results, left_results, right_results


if __name__ == '__main__':
    reader = easyocr.Reader(['en'])
    base_path = '/home/hdd1/wanghaoran/sift/data/ocr/'
    files = glob(os.path.join(base_path, '1.jpg'))
    for f in files:
        result, left, right = lr90_ocr(reader, f)
        total = result + left + right
        print('=======================================')
        print(f)
        print(result)
        print(haveMerge(result))
        for s in result:
            if haveMerge([s]):
                print(s)
