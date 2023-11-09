import os
import shutil
from glob import glob

target_base_path = '/home/hdd1/wanghaoran/sift/sample/0320'
txt_file = '/home/hdd1/wanghaoran/sift/diff/2022/0320/v1_v3/v3_189.txt'

def copy_file(src_path):
    tail = '/'.join(src_path.split('/')[-4:])
    target = os.path.join(target_base_path, tail)
    dir = '/'.join(target.split('/')[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    shutil.copy(src_path, target)


if __name__ == '__main__':
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            file_path = line.strip('\n')
            full_pic_path = glob('_'.join(file_path.split('_')[:-1])+'_full*')[0]
            ori_pic_path = glob('_'.join(file_path.split('_')[:-1])+'.*')[0]
            copy_file(file_path)
            copy_file(full_pic_path)
            copy_file(ori_pic_path)